OS=ubuntu20.04

# Install Docker, see https://docs.docker.com/engine/install/ubuntu/
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Install Habana, see https://docs.habana.ai/en/latest/Installation_Guide/Base_OS_AMI_AWS.html#run-using-containers
curl -X GET https://vault.habana.ai/artifactory/api/gpg/key/public | sudo apt-key add --
echo "deb https://vault.habana.ai/artifactory/debian focal main" >> /etc/apt/sources.list.d/artifactory.list
sudo dpkg --configure -a
sudo apt update
sudo apt install -y habanalabs-firmware
sudo apt install -y habanalabs-dkms
sudo modprobe habanalabs_en
sudo modprobe habanalabs
sudo apt install -y habanalabs-container-runtime
sudo mkdir -p /etc/docker && sudo cp tests/ci/habana_docker_daemon.json /etc/docker/daemon.json
sudo systemctl restart docker
