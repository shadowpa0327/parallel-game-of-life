FROM nvidia/cudagl:11.4.2-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

RUN apt-get update && apt-get install -y \
    sudo \
    python3 git-core gnupg flex bison build-essential zip curl zlib1g-dev gcc-multilib g++-multilib libc6-dev-i386 libncurses5 lib32ncurses5-dev x11proto-core-dev libx11-dev lib32z1-dev libgl1-mesa-dev libxml2-utils xsltproc unzip fontconfig cmake freeglut3 freeglut3-dev libglm-dev libglew-dev\
  && rm -rf /var/lib/apt/lists/*

ARG WORKSPACE_DIR="/ws"
ENV HOME_PATH="${WORKSPACE_DIR}/home"

RUN mkdir -p "${WORKSPACE_DIR}"

RUN curl -o repo https://storage.googleapis.com/git-repo-downloads/repo \
    && install -m 755 repo /usr/bin/repo \
    && rm -f repo \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 2

RUN groupadd --gid 1000 user \
  && useradd --uid 1000 --gid user --shell /bin/bash --create-home user --home-dir "${HOME_PATH}" \
  && mkdir -p /etc/sudoers.d \
  && echo 'user ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers.d/user \
  && chmod 440 /etc/sudoers.d/user

COPY ./docker-entrypoint.sh /bin/docker-entrypoint.sh
ENTRYPOINT ["docker-entrypoint.sh"]

WORKDIR "$WORKSPACE_DIR"
CMD ["bash"]

