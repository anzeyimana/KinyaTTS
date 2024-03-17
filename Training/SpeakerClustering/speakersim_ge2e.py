from __future__ import print_function, division

import warnings

warnings.filterwarnings("ignore")

import sys

import random
import progressbar
from datetime import datetime

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


SPLITS_DIR = 'SPLITS/'

def model_parameter_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def read_lines(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    lines = [line.rstrip('\n') for line in f]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    f.close()
    return lines
def write_lines(lines, file_name):
    f = open(file_name, 'w', encoding='utf-8')
    for l in lines:
        f.write(l+'\n')
    f.close()

def time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def date_now():
    return datetime.now().strftime("%Y-%m-%d")

def get_centroids_prior(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid = centroid + utterance
        centroid = centroid/len(speaker)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids

def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids

def get_centroid(embeddings, speaker_num, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid/(len(embeddings[speaker_num])-1)
    return centroid

def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance

    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids

def get_cossim_prior(embeddings, centroids):
    # Calculates cosine similarity matrix. Requires (N, M, feature) input
    cossim = torch.zeros(embeddings.size(0),embeddings.size(1),centroids.size(0))
    for speaker_num, speaker in enumerate(embeddings):
        for utterance_num, utterance in enumerate(speaker):
            for centroid_num, centroid in enumerate(centroids):
                if speaker_num == centroid_num:
                    centroid = get_centroid(embeddings, speaker_num, utterance_num)
                output = F.cosine_similarity(utterance,centroid,dim=0)+1e-6
                cossim[speaker_num][utterance_num][centroid_num] = output
    return cossim

def get_cossim(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1],
        -1
    )
    embeddings_flat = embeddings.view(
        embeddings.shape[0] * num_utterances,
        -1
    )
    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1]
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0)
    )
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff

def calc_loss_prior(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))
    for j in range(len(sim_matrix)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum()+1e-6).log_()))
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss

def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss

class SpeechEmbedder(nn.Module):

    def __init__(self, nmels=80, num_layer=3, hidden=768, proj=256):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(nmels, hidden, num_layers=num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hidden, proj)

    def forward(self, x): # x: (N,L,H_in); H_in = n_mels
        x, _ = self.LSTM_stack(x.float())  # x: (N,L,H_out)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x # x: (N,E)


class GE2ELoss(nn.Module):

    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w * cossim.to(self.device) + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss

    def similarity(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w * cossim.to(self.device) + self.b
        return sim_matrix



class SampleConfig:
    def __init__(self):
        self.resamplers = {}
        self.mel_spectrograms = {}
        self.resample_rate = 16000
        self.lowpass_filter_width = 64
        self.rolloff = 0.9475937167399596
        self.resampling_method = "kaiser_window"
        self.beta = 14.769656459379492
        self.n_fft = 1024
        self.n_mels = 80

def get_sampling_tools(cfg: SampleConfig, sample_rate, waveform):
    sr_kwd = str(sample_rate)
    if sr_kwd in cfg.resamplers:
        resampler = cfg.resamplers[sr_kwd]
        mel_spectrogram = cfg.mel_spectrograms[sr_kwd]
    else:
        win_length = cfg.resample_rate * 25 // 1000  # 25ms
        hop_length = cfg.resample_rate * 10 // 1000  # 10ms
        resampler = T.Resample(sample_rate, cfg.resample_rate, lowpass_filter_width=cfg.lowpass_filter_width,
                                 rolloff=cfg.rolloff, resampling_method=cfg.resampling_method,
                                 dtype=waveform.dtype,
                                 beta=cfg.beta, )  # .cuda()
        mel_spectrogram = T.MelSpectrogram(sample_rate=cfg.resample_rate, n_fft=cfg.n_fft,
                                             win_length=win_length,
                                             hop_length=hop_length, center=True, pad_mode="reflect", power=2.0,
                                             norm="slaney", onesided=True, n_mels=cfg.n_mels,
                                             mel_scale="htk", )  # .cuda()
        cfg.resamplers[sr_kwd] = resampler
        cfg.mel_spectrograms[sr_kwd] = mel_spectrogram
    return resampler, mel_spectrogram


def form_input_data(waveform, sample_rate, sample_config, log_eps=1e-36):
    resampler, mel_spectrogram = get_sampling_tools(sample_config, sample_rate, waveform)
    with torch.no_grad():
        if int(sample_rate) != int(sample_config.resample_rate):
            waveform = resampler(waveform)
        waveform = torch.mean(waveform, 0, keepdim=False)
        input_data = mel_spectrogram(waveform)  # (F,L)
        input_data = torch.log(input_data + log_eps)
    return input_data # (F,L)

class SpeechEmbedDataset(Dataset):
    def __init__(self, splits_tsv ='SPLITS.tsv',
                 num_speakers_per_batch = 4,
                 num_utterances_per_batch = 5,
                 num_batches = 1000):
        self.num_speakers_per_batch = num_speakers_per_batch
        self.num_utterances_per_batch = num_utterances_per_batch
        self.num_batches = num_batches
        self.splits_tsv = splits_tsv
        self.sample_config = SampleConfig()
        self.batches = []
        self.reload()

    def reload(self):
        print(time_now(), 'SpeechEmbedDataset: Reloading items ...')
        self.batches.clear()
        self.batches = []
        lines = read_lines(self.splits_tsv)
        splits = [l.split('\t')[8:] for l in lines]
        splits = [p for p in splits if len(p) > (self.num_utterances_per_batch * 2)]
        for _ in range(self.num_batches):
            batch = []
            speakers = random.sample(splits,self.num_speakers_per_batch)
            for utterances in speakers:
                batch.extend(random.sample(utterances,self.num_utterances_per_batch))
            self.batches.append(batch)
        print(time_now(), f'SpeechEmbedDataset: Got {len(self.batches)} items!')

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        if(idx == 0):
            self.reload()
        batch = self.batches[idx]
        x_seq = []
        x_lengths = []
        for file in batch:
            x, sr = torchaudio.load(SPLITS_DIR + file)
            x = form_input_data(x, sr, self.sample_config)
            x_lengths.append(x.size(1))
            x_seq.append(x.transpose(0, 1))
        x_data = pad_sequence(x_seq, batch_first=True) # (N,L,H_in)
        return x_data[:,:min(x_lengths),:] #pack_padded_sequence(x_data, torch.tensor(x_lengths), batch_first=True, enforce_sorted=False)

def ge2e_collate_fn(items):
    return items[0]

def flush_stdout(bar):
    bar.fd.flush()
    sys.stdout.flush()
    sys.stderr.flush()

def train_one_epoch(data_loader, embedder_net:SpeechEmbedder, ge2e_loss:GE2ELoss, device,
                   optimizer: torch.optim.SGD,
                   num_speakers_per_batch = 4,
                   num_utterances_per_batch = 5):
    embedder_net.train()
    ge2e_loss.train()
    total_loss = 0.0
    count = 0.0
    for batch_idx, batch_data_item in enumerate(data_loader):
        embeddings = embedder_net(batch_data_item.to(device)) # (Batch, Embedding)
        loss = ge2e_loss(embeddings.view(num_speakers_per_batch, num_utterances_per_batch, embeddings.size(-1)))  # wants (Speaker, Utterances, Embedding)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
        torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        count += 1.0
    return total_loss/count

def train_loop(lr=0.01,
               num_speakers_per_batch = 4,
               num_utterances_per_batch = 5,
               num_epochs = 1000,
               num_batches = 1000,
               save_path='models/speech_embed_ge2e.pt'):
    device = torch.device('cuda:0')
    embedder_net = SpeechEmbedder().to(device)
    ge2e_loss = GE2ELoss(device)
    print('Training speech embedding model with parameter count =',
          model_parameter_count(embedder_net), '+',
          model_parameter_count(ge2e_loss))
    optimizer = torch.optim.SGD([{'params': embedder_net.parameters()}, {'params': ge2e_loss.parameters()}], lr=lr)
    dataset = SpeechEmbedDataset(splits_tsv ='SPLITS.tsv',
                                 num_speakers_per_batch = num_speakers_per_batch,
                                 num_utterances_per_batch = num_utterances_per_batch,
                                 num_batches = num_batches)
    data_loader = DataLoader(dataset, collate_fn=ge2e_collate_fn, batch_size=1, pin_memory=True, drop_last=False, shuffle=False, num_workers=2, persistent_workers=True)
    with progressbar.ProgressBar(initial_value=0, max_value=num_epochs, redirect_stdout=True) as bar:
        for epoch in range(num_epochs):
            bar.update(epoch)
            flush_stdout(bar)
            embedder_net = embedder_net.to(device)
            ge2e_loss = ge2e_loss.to(device)
            total_loss = train_one_epoch(data_loader, embedder_net, ge2e_loss, device,
                                         optimizer,
                                         num_speakers_per_batch = num_speakers_per_batch,
                                         num_utterances_per_batch = num_utterances_per_batch)
            print(f'{time_now()} Epoch: {epoch+1} Loss: {total_loss:.4f}')
            embedder_net = embedder_net.cpu()
            ge2e_loss = ge2e_loss.cpu()
            embedder_net.eval()
            ge2e_loss.eval()
            torch.save({'embedder_net':embedder_net.state_dict(),
                        'ge2e_loss':ge2e_loss.state_dict()},
                       save_path)
def eval_similarity(sample_config, embedder_net, device, files):
    x_seq = []
    x_lengths = []
    for file in files:
        x, sr = torchaudio.load(file)
        x = form_input_data(x, sr, sample_config)
        x_lengths.append(x.size(1))
        x_seq.append(x.transpose(0, 1))
    x_data = pad_sequence(x_seq, batch_first=True)  # (N,L,H_in)
    batch_data_item = x_data[:, :min(x_lengths), :]
    with torch.no_grad():
        embeddings = embedder_net(batch_data_item.to(device))
        sim = get_cossim(embeddings.view(len(files), 1, embeddings.size(-1)), embeddings)
    return sim.squeeze().numpy(force=True)

def eval_huza_imvugo_speech_similarity(model_file):
    sample_config = SampleConfig()
    device = torch.device('cuda:0')
    embedder_net = SpeechEmbedder().to(device)
    state_dict = torch.load(model_file, map_location={'cuda:0':'cuda:0'})
    embedder_net.load_state_dict(state_dict['embedder_net'])
    embedder_net.eval()
    print('Model ready!\n')
    lines = read_lines('huza_imvugo_speech_all.txt')
    src = dict()
    for line in lines:
        key = line.split('_')[0]
        if key in src:
            src[key] = src[key] + [line]
        else:
            src[key] = [line]
    keys = [k for k in src]
    samples = [sorted(src[k], key=lambda x:(int(x.split('_')[-3])-int(x.split('_')[-4])), reverse=True)[0] for k in keys]
    itr = 0
    n = len(samples)
    total = int(((n*n)-n)/2)
    print(f'Evaluating {n} files: {total} pairs!\n')
    with open('huza_imvugo_speech_similarity.tsv', 'a', encoding='utf-8') as out_file:
        with progressbar.ProgressBar(initial_value=0, max_value=total, redirect_stdout=True) as bar:
            for a in range(n):
                for b in range(a + 1, n):
                    if (itr%1000) == 0:
                        bar.update(itr)
                        flush_stdout(bar)
                    files = [f'mp3/{samples[a]}.mp3', f'mp3/{samples[b]}.mp3']
                    sim = eval_similarity(sample_config, embedder_net, device, files)
                    y = sim[0, 1]
                    out_file.write(f'{samples[a]}\t{samples[b]}\t{y:.2f}\n')
                    out_file.flush()
                    itr += 1


def eval_loop(model_file):
    sample_config = SampleConfig()
    device = torch.device('cuda:0')
    embedder_net = SpeechEmbedder().to(device)
    state_dict = torch.load(model_file, map_location={'cuda:0':'cuda:0'})
    embedder_net.load_state_dict(state_dict['embedder_net'])
    embedder_net.eval()
    print('Model ready!\n')
    while True:
        file1 = input("\nAudio file 1: ")
        if file1 in {'exit', 'EXIT', 'e', 'E', 'quit', 'QUIT', 'q', 'Q'}:
            print('Exiting ...')
            sys.exit(0)
        file2 = input("\nAudio file 2: ")
        if file2 in {'exit', 'EXIT', 'e', 'E', 'quit', 'QUIT', 'q', 'Q'}:
            print('Exiting ...')
            sys.exit(0)
        try:
            sim = eval_similarity(sample_config, embedder_net, device, [file1, file2])
            y = sim[0,1]
            # print(sim, flush=True)
            print(f'\nSimilarity: {y:.2f}\n', flush=True)
        except Exception as e:
            print(e)


if __name__ == '__main__':
    # train_loop(lr=0.01, num_speakers_per_batch=4, num_utterances_per_batch=5, num_epochs=1000, num_batches=1000,
    #            save_path='models/speech_embed_ge2e_2023-11-29_1000ep_x_1000ba_.pt')
    # eval_loop('models/speech_embed_ge2e_2023-11-29_1000ep_x_1000ba_.pt')
    eval_huza_imvugo_speech_similarity('models/speech_embed_ge2e_2023-11-29_1000ep_x_1000ba_.pt')
