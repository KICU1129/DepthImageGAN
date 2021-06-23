# DepthImageGAN

## **Docker環境 構築**

**1. 取り合えずpytorch をpull**
~~~ 
docker pull pytorch/pytorch
~~~

---
**2. ImageとContainerの作成**
~~~
docker compose up -d --build
~~~

**GPU ver**
~~~
docker build -t [IMAGE / NAME] .
~~~
---

**3. Containerへ接続する**
~~~
docker compose exec "CONTAINER COMMAND" bash
~~~

**GPU ver**
~~~
docker run --rm --user $UID -v $PWD:$PWD -w $PWD  --gpus all -it [IMAGE NAME] bash
~~~

---
  
**4. Containerから切断**
~~~
docker compose down
~~~
---

**5. Containerへの再接続**
~~~
docker compose up -d
~~~

GPUの場合3のGPU verをもう一回やればおｋ
 
---

**6. いらないイメージの削除**
~~~
docker image rm "IMAGE ID" -f
~~~



## **References**
1. [ついにWSL2+docker+GPUを動かせるようになったらしいので試してみる](https://qiita.com/yamatia/items/a70cbb7d8f5101dc76e9)
2. [DockerでPython実行環境を作ってみる](https://qiita.com/jhorikawa_err/items/fb9c03c0982c29c5b6d5)
3. [DockerでのディープラーニングGPU学習環境構築方法](https://qiita.com/karaage0703/items/e79a8ad2f57abc6872aa)
