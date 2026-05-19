#!/bin/bash
OS="$(uname -s)"

# OS dependent installations
case "$OS" in
  Linux)
    sudo apt install -y cpanminus
    ;;
  Darwin)
    brew install cpanminus
    ;;
  *)
    curl -L https://cpanmin.us | perl - App::cpanminus
    ;;
esac

cpanm --installdeps .
pip install -r requirements.txt