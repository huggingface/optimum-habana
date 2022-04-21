OS=ubuntu20.04

curl -X GET https://vault.habana.ai/artifactory/api/gpg/key/public | sudo apt-key add --
echo "deb https://vault.habana.ai/artifactory/debian focal main" >> /etc/apt/sources.list.d/artifactory.list
sudo dpkg --configure -a
sudo apt update
sudo apt install -y habanalabs-firmware
sudo apt install -y habanalabs-dkms
sudo modprobe habanalabs_en
sudo modprobe habanalabs
sudo apt install -y habanalabs-container-runtime
sudo cp daemon_ci.json /etc/docker/daemon.json
sudo systemctl restart docker
