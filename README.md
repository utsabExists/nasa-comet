# nasa-comet

# Imp Links :
1. https://www.topcoder.com/challenges/fcad16e0-9ca6-4510-8bd9-af3ed026b140
2. https://github.com/topcoderinc/marathon-docker-template/tree/master/data-plus-code-style

# Docker Links :
1. https://www.docker.com/why-docker

# fits header :
https://lasco-www.nrl.navy.mil/index.php?p=content/keywords

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

# Open the fts files using :
1. C:\Users\utbose\Downloads\MicroObservatoryImageWindows2.3\run.bat

# Steps and plan for the algorithm
1. First verify that the processed image can identify all the TP points w.r.t the ground truth.
2. If everything works well above, calculate the min and max idff between two points in ground truth
3. Use above to filter out all "comet like patterns" from the image sequence of each comet.
4. Super impose all comet like patterns into 1 quadrant (ignore those which are too long and extends over more than 1 quad for now)
5. Create new images/training set with comet like patterns and label each known (ground thruth pattern) as "comet" and other as "not comet"
6. Build and ML model with the pre processed training set
7. Follow steps 2 to 7 for test data and generate a test set
8. Run test set with ML model from 6 and identfiy all comets
9. Post process each comet identified image and pot their original co-ordinates in solution file.