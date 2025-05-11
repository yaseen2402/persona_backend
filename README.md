## docker build -t chatai .

the -t flag stands for "tag".

It's used to give a name and optionally a tag in the name:tag format to the Docker image you are building.

chatai: This is the name given to the image. You can use this name to refer to the image later, for example, when running a container based on this image.

## docker run -p 5000:5000 chatai