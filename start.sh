#!/bin/bash

# 환경 변수를 활용한 설정
# ENV 값 확인 (기본값: development)
ENV=${ENV:-"development"}
echo "Running in $ENV environment"

# 환경 변수를 이용해 Python 스크립트로 설정 로드
# environments.py 파일의 설정을 활용합니다
HOST=$(python3 -c "from config.environments import current_env; print(current_env['HOST'])")
PORT=$(python3 -c "from config.environments import current_env; print(current_env['PORT'])")
DEBUG=$(python3 -c "from config.environments import current_env; print(str(current_env.get('DEBUG', False)).lower())")

echo "Starting server on $HOST:$PORT (debug: $DEBUG)..."

# Uvicorn 서버 실행 - reload 옵션을 조건부로 설정
if [ "$DEBUG" = "true" ]; then
    exec uvicorn main:app --host $HOST --port $PORT --reload
else
    exec uvicorn main:app --host $HOST --port $PORT
fi 