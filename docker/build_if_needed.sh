#!/bin/zsh

git fetch
if [ $(git rev-parse HEAD) == $(git rev-parse @{u}) ]
then
echo no pull
else
echo pull
fi
