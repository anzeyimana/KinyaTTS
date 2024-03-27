import torchmetrics

if __name__ == '__main__':
    import progressbar
    from deepkin.models.kinspeak_ctc_decode import norm_text_eval
    import requests
    f = open('tts_eval_content.txt', 'r', encoding='utf-8')
    lines = [l.rstrip('\n') for l in f if len(l)>10]
    f.close()
    predictions = []
    targets = []
    with progressbar.ProgressBar(initial_value=0,
                                 max_value=len(lines),
                                 redirect_stdout=True) as bar:
        for itr,AUDIO_TRANSCRIPT in enumerate(lines):
            bar.update(itr)
            try:
                response = requests.post('http://127.0.0.1:9696/tts',
                                         json={'input': AUDIO_TRANSCRIPT, 'output': '/home/user/test_tts.wav'})
                if response.status_code == 200:
                    response = requests.post('http://127.0.0.1:9696/asr',
                                             json={'input': '/home/user/test_tts.wav'})
                    if response.status_code == 200:
                        js = response.json()
                        TEXT_PREDICTION = js['output']
                        cer, wer, pred_txt, target_txt = norm_text_eval(TEXT_PREDICTION, AUDIO_TRANSCRIPT)
                        predictions.append(pred_txt)
                        targets.append(target_txt)

            except:
                pass
    print('==> Final overall CER/WER [%]:', '{:.1f}/{:.1f}'.format(100.0 * torchmetrics.functional.char_error_rate(preds=predictions, target=targets),
                                                                   100.0 * torchmetrics.functional.word_error_rate(preds=predictions, target=targets)), flush=True)
