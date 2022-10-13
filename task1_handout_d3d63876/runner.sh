docker build --tag task1 .
#docker run --rm -v "$( cd "$( results "$0" )" && pwd )":/results task1
docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task1
