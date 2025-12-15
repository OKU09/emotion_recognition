# はじめに

メディア系演習の感情認識です

conda で環境構築することを強く推奨します．

録音データから感情を読み取るのは hume_batch.py を，リアルタイムで感情を読み取るのは hume_websoket.py を使用してください．
どちらも hume ai の API キーが必要です．(自身でコードに上書きで入力してください．書いてある API キーは使えません．)

テキストから感情認識を行う場合は kan.py を使用してください．
追加で vosk-model-small-ja-0.22 モデルが必要です．さらに，kanjou-model ファイルを作成し，added_tokens.json, config.json, entity_vocab.json, pytorch_model.bin, sentencepiece.bpe.model, special_tokens_map.json, tokenizer_config.json を中に入れて下さい．

表情から感情を認識する場合は image-emotion.py を使用してください．
事前に準備するものやコードに書き加えることはありませんが，それぞれのライブラリ，モデルのインストールに時間がかかります．conda で環境構築したものは動作していますが，venv では動作を確認できませんでした．

以下にdiscordなどをもとにした実行手順を示します．

# 実行手順

## PART0 仮想環境構築(conda)

1. まずは anaconda をダウンロードします.
```Powershell
Invoke-WebRequest -Uri "https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Windows-x86_64.exe" -outfile ".\Downloads\Anaconda3-2025.06-0-Windows-x86_64.exe"
```
2. ダウンロードしたインストーラを実行します
```Powershell
Start-Process -FilePath ".\Downloads\Anaconda3-2025.06-0-Windows-x86_64.exe"
```
3. インストーラの指示に沿ってインストールを完了してください．
4. 以下のコマンドが正しく動作すれば成功です．
```Poershell
conda info
```
4*. powershell等でエラーが出た場合，再起動を行ってください．それでも直らない場合は `anaconda pronpt`を使用してください．(anacondaがインストールされていればスタートメニューの検索で出てくると思います．)  
  
5. python3.11.3の環境を作ります．
```Powershell
conda create -n py311 python=3.11.3
```
6. 作成した環境をactivateします．
```Powershell
conda activate py311
```
6*. 今後は該当ファイルで6.を実行すれば大丈夫です．

## PART1 音声分野

1. [hume](https://www.hume.ai/) に会員登録をしてAPIキーを取得してください．
2. github上にあるhume_websoket.pyをダウンロードしてください．
3. hume_websocket.py内部のAPIキーを取得したものに書き換えてください。(元々書いてあるものは無効なキーです。)
4. hume_websocket.pyを実行します。初回は足りないライブラリ(websocket,sounddevice...)があるためエラーが出てくると思います。適宜ライブラリをインストールし、実行し直してください。
```Powershell
pip install ○○
```
もしくは、
```Powershell
python -m pip install ○○
```
5. 実行できたら成功です。声が検出されれば感情を出力します。また、hume apiは利用時間で料金がかかります。会員登録で20ドル分クレジットが貰えるのでそれを超えないように気をつけてください。残りのクレジットはhumeマイページのbillingのexpression measurementのタブにあります。
