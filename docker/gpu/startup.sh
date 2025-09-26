#!/bin/bash
for i in {0..7}; do
  PORT=$((8880 + i))
  docker compose run -d -p ${PORT}:8880 kokoro-tts
done