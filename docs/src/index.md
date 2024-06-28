# CYAxiverse.jl

A [Julia](https://julialang.org) package to compute axion/ALP spectra from string theory (using output of [CYTools](https://cytools.liammcallistergroup.com/))

---

## Authors ✒️
[Viraf M. Mehta](https://inspirehep.net/authors/1228975)

## Installation 💾 
!!! warning
    This package is currently _not_ registered with the `julia` package manager and is still under development.  **Use at your own risk!**

!!! info
    Currently this package runs in a docker container with `CYTools`.  In time, the installation process will be automated.

!!! warning
    The Docker container is just over 3GB

!!! tip
    There are a couple of stages that take around 10-15 minutes to complete and so be ready with the kettle ☕

To build this docker container, follow these instructions (currently only appropriate for UNIX-based systems):
    
- install the appropriate [Docker Desktop](https://docs.docker.com/desktop/) for your system
- in a terminal, create a new directory for `CYTools` and `CYAxiverse` e.g.
```
export CYAXIVERSE_ROOT=$HOME/cyaxiverse &&
export CYAXIVERSE_REPO=$CYAXIVERSE_ROOT/CYAxiverse_repo &&
export CYTOOLS_REPO=$CYAXIVERSE_ROOT/CYTools_repo &&
mkdir $CYAXIVERSE_ROOT &&
mkdir $CYTOOLS_REPO && 
mkdir $CYAXIVERSE_REPO
```
!!! tip
    Change `$HOME → /root/path/where/cyaxiverse/will/live` in the first line and it should propagate through

!!! warning
    A trailing `/` will break this, be careful.

- clone the `CYTools` repository
```
cd $CYTOOLS_REPO &&
git clone https://github.com/LiamMcAllisterGroup/cytools.git
```
- clone[^1] this repository (currently `vmm` branch is up-to-date)

[^1]: 
    one can also `git pull` the repository -- this would enable the `CYAxiverse.jl` package to be updated (while under development) with specific directory binding.  Use this command instead:
    ```
    mkdir $CYAXIVERSE_REPO/CYAxiverse.jl &&
    cd $CYAXIVERSE_REPO/CYAxiverse.jl &&
    git init &&
    git pull https://github.com/Julia-meets-String-Theory/CYAxiverse.jl.git vmm 
    ```
    and then you can keep this up-to-date as improvements are pushed with 
    ```
    git pull https://github.com/Julia-meets-String-Theory/CYAxiverse.jl.git vmm
    ```

```
cd $CYAXIVERSE_REPO && 
git clone -b vmm https://github.com/Julia-meets-String-Theory/CYAxiverse.jl.git
```
- replace the default `Dockerfile` in your `CYTools` directory with the `Dockerfile` in `$CYAXIVERSE_REPO` and move `add_CYAxiverse.jl` there too, _e.g._
```
mv $CYTOOLS_REPO/cytools/Dockerfile $CYAXIVERSE_ROOT/Dockerfile_CYTools && 
cp $CYAXIVERSE_REPO/CYAxiverse.jl/Dockerfile $CYTOOLS_REPO/cytools/ && 
cp $CYAXIVERSE_REPO/CYAxiverse.jl/add_CYAxiverse.jl $CYTOOLS_REPO/cytools/
```
- run the following command from your `CYTools` directory _e.g._ `cyaxiverse/cytools/` :
```
docker build --no-cache --force-rm -t cyaxiverse:uid-$UID --build-arg USERNAME=cytools \
--build-arg USERID=$UID --build-arg ARCH=amd64 \
--build-arg AARCH=x86_64 --build-arg VIRTUAL_ENV=/home/cytools/cytools-venv/ \
--build-arg ALLOW_ROOT_ARG=" " --build-arg PORT_ARG=$(($UID+2875)) .
```
!!! note 
    This takes ~15 minutes on a
    ``` 
    MacBook Pro (13-inch, 2017, Two Thunderbolt 3 ports)
    Processor   2,3 GHz Dual-Core Intel Core i5
    Memory      16 GB 2133 MHz LPDDR3
    ```
    so make yourself a cup of tea 😃
- create a `dir` for your data _e.g._
```
export CYAXIVERSE_DATA=$CYAXIVERSE_ROOT/CYAxiverse_database &&
mkdir $CYAXIVERSE_DATA
```
- you can now run your docker image with[^2]

[^2]: 
    in order to keep `CYAxiverse.jl` up-to-date (while under development), bind the local `CYAxiverse` version with the `CYAxiverse.jl` directory in the Docker container, _e.g._ with the
    ```
    --mount type=bind,source="$CYAXIVERSE_REPO/CYAxiverse.jl",target=/opt/CYAxiverse.jl,readonly
    ```
    option included, _i.e._ 
    ```
    docker container run -it --mount type=bind,source=$CYAXIVERSE_DATA,target=/database\
    --mount type=bind,source=$CYAXIVERSE_REPO/CYAxiverse.jl,target=/opt/CYAxiverse.jl\
    -p 8994:8996 cyaxiverse:uid-$UID
    ```
    Enabling this ensures the `CYAxiverse.jl` version compiled in the Docker container matches the one most recently `pull`ed from the repository.

```
docker container run -it --mount type=bind,source=$CYAXIVERSE_DATA,target=/database\
-p 8994:8996 cyaxiverse:uid-$UID
```
If this is the first run, `julia` will precompile the required packages for `CYAxiverse.jl` which, at the moment, takes about 5 minutes.  Then, opening a browser and going to [`http://localhost:8994`](http://localhost:8994), you will be presented with the [`Pluto`](https://github.com/fonsp/Pluto.jl/wiki) notebook interface.  You can save your new notebook in `/opt/CYAxiverse/notebooks`.

Enjoy! 
![:deploy_parrot:](https://emoji.slack-edge.com/T7DMEKZMH/deployparrot/ef6c902688cec864.gif)


## Acknowledgements 🙏
This project was born after publication of [Superradiance in String Theory](https://iopscience.iop.org/article/10.1088/1475-7516/2021/07/033) and I am grateful to my collaborators for their input while this code was evolving.  Huge thanks also to the authors of [CYTools](https://cy.tools/) for their ongoing hard work and to [Mona Dentler](https://inspirehep.net/authors/1635411) for constant help throughout the development of this package.
