OS=ubuntu20.04

# Install Docker
sudo apt update
sudo apt-get install ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io

# Install Habana
curl -X GET https://vault.habana.ai/artifactory/api/gpg/key/public | sudo apt-key add --
echo "deb https://vault.habana.ai/artifactory/debian focal main" >> /etc/apt/sources.list.d/artifactory.list
sudo dpkg --configure -a
sudo apt update
sudo apt install -y habanalabs-firmware
sudo apt install -y habanalabs-dkms
sudo modprobe habanalabs_en
sudo modprobe habanalabs
sudo apt install -y habanalabs-container-runtime
sudo mkdir -p /etc/docker && sudo cp daemon_ci.json /etc/docker/daemon.json
sudo systemctl restart docker
