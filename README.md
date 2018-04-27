# DPSEGNET
Novel approach to image segmentation: Segnet + hebbian

#### [Segnet](https://arxiv.org/abs/1511.00561)
![alt text](https://saytosid.github.io/images/segnet/Complete%20architecture.png)

#### [Hebbian](https://arxiv.org/abs/1804.02464)
![alt text](https://github.com/kho226/DPSEGNET/blob/master/images/hebbian.png)

#### To - Do
- [ ] add documentation
- [ ] implement multi-threaded data augmentation
- [ ] pass one image through the network!

#### Installation (macOS Sierra Version 10.12.1)
```
sudo easy_install pip
sudo pip install virtualenv
git clone https://github.com/kho226/DPSEGNET
cd <project_path>
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```