#!/usr/bin/bash

apt update
echo "--> Ruby"
apt install -y ruby-full

echo "--> PHP"
apt install -y software-properties-common ca-certificates lsb-release apt-transport-https
add-apt-repository ppa:ondrej/php
apt update -y
apt install -y php-{pear,cgi,common,curl,mbstring,gd,bcmath,json,xml,fpm,intl,zip} php8.0


echo "--> JavaScript"
apt install curl
curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash
# Check if the lines containing NVM_DIR already exist in .bashrc
if ! grep -q 'NVM_DIR' ~/.bashrc; then
  echo "# --- NVM ---" >> ~/.bashrc
  grep 'NVM_DIR' ~/.zshrc >> ~/.bashrc
fi
PS1=1 source ~/.bashrc
apt install npm
nvm install 20.17.0
node -e "console.log('Running Node.js ' + process.version)"
npm i -g npm
npm install -g lodash
npm i --save lodash


echo "--> TypeScript"
npm install -g typescript



