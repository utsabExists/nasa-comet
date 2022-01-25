# nasa-comet

# Imp Links :
1. https://www.topcoder.com/challenges/fcad16e0-9ca6-4510-8bd9-af3ed026b140
2. https://github.com/topcoderinc/marathon-docker-template/tree/master/data-plus-code-style

# Docker Links :
1. https://www.docker.com/why-docker

# Instructions to run using docker (in windows) :
1. docker build -t docker-cometproject .
2. docker run -t docker-cometproject
3. From CLI terminal (open docker desktop and click on CLI icon for this container)
4. ./train.sh data/train
5. ./test.sh data/test solution.csv
6. Copy the file from docker to host using the following commands
    1. docker ps
    2. Get the container id from above command
    3. docker cp <container-id>:/work/sloution.csv <hostmachine-path>