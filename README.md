# classify_pill_AI
[경구약제 이미지 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=576)    
python 3.8

### package
```
pip3 install torch torchvision
pip3 install numpy​
pip3 install opencv-python​
pip3 install imgaug​
pip3 install PIL ​
pip3 install tqdm​
pip3 install codecs​
pip3 install json ​
pip3 install matplotlib​
```


```
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
sudo apt-get install software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable edge"

sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli

sudo docker version

sudo sudo docker load -i pill_class.tar
sudo docker run -it -v /home/favorcat/proj/proj_pill:/home/favorcat/proj/proj_pill ubuntu:pill