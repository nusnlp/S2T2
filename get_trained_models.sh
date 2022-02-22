#!/usr/bin/env bash

MODEL_DIR="trained_models"
mkdir -p $MODEL_DIR

cd $MODEL_DIR
mkdir -p LIF en jp

#LIF
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ar2lE9wCLwh2giosmW_2jDhD1RG-I5Zx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ar2lE9wCLwh2giosmW_2jDhD1RG-I5Zx" -O LIF/model.pt.tar && rm -rf /tmp/cookies.txt
#en
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PcSrGduVfUzEMbGyxTlM3tWBgB1mUN_j' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1PcSrGduVfUzEMbGyxTlM3tWBgB1mUN_j" -O en/model.pt.tar && rm -rf /tmp/cookies.txt
#jp
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1gFSXvmuwxhuemcdkwqUQ8OVkCF7wrvMM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1gFSXvmuwxhuemcdkwqUQ8OVkCF7wrvMM" -O jp/model.pt.tar && rm -rf /tmp/cookies.txt

cd ..