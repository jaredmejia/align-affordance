set -x 
set -e


# Install Frankmocap
rm -r third_party/frankmocap
mkdir -p third_party
# my modification on relative path
git clone https://github.com/judyye/frankmocap.git third_party/frankmocap
cd third_party/frankmocap
bash scripts/install_frankmocap.sh
cd ../..


# # detectron2, requried by frankmocap
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html


# pytorch3d for rendering
pip install git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2
