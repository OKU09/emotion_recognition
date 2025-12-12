# はじめに

メディア系演習の感情認識です

conda で環境構築することを強く推奨します．

録音データから感情を読み取るのは hume_batch.py を，リアルタイムで感情を読み取るのは hume_websoket.py を使用してください．
どちらも hume ai の API キーが必要です．(自身でコードに上書きで入力してください．書いてある API キーは使えません．)

テキストから感情認識を行う場合は kan.py を使用してください．
追加で vosk-model-small-ja-0.22 モデルが必要です．さらに，kanjou-model ファイルを作成し，added_tokens.json, config.json, entity_vocab.json, pytorch_model.bin, sentencepiece.bpe.model, special_tokens_map.json, tokenizer_config.json を中に入れて下さい．

表情から感情を認識する場合は image-emotion.py を使用してください．
事前に準備するものやコードに書き加えることはありませんが，それぞれのライブラリ，モデルのインストールに時間がかかります．conda で環境構築したものは動作していますが，venv では動作を確認できませんでした．

# 実行手順

## PART0 仮想環境構築

1. まずは anaconda にアクセスします．
