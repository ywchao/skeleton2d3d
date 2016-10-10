# skeleton2d3d

## Dependencies

Make sure the following are installed.

- [Torch7](https://github.com/torch/distro)
- [matio-ffi](https://github.com/soumith/matio-ffi.torch)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5) (only needed for Penn Action)

## Setup

0. Download and extract the Human3.6M dataset from `https://vision.imar.ro/human3.6m`. Only the following files are needed.
  ```Shell
  Poses_RawAngles_S1.tgz
  Poses_RawAngles_S5.tgz
  Poses_RawAngles_S6.tgz
  Poses_RawAngles_S7.tgz
  Poses_RawAngles_S8.tgz
  Poses_RawAngles_S9.tgz
  Poses_RawAngles_S11.tgz
  ```

0. Clone the skeleton2d3d epository.
  ```Shell
  git clone git@git.corp.adobe.com:chao/image-play.git
  ```

0. Create symlinks for the downloaded Human3.6M dataset. `$H36M_ROOT` should contain `S1`, `S5`, `S6`, `S7`, `S8`, `S9`, and `S11`.
  ```Shell
  cd $S2D3D_ROOT
  ln -s $H36M_ROOT ./external/Human3.6M
  ```

0. Generate meta files. Start MATLAB `matlab` under `skeleton2d3d`. You should see the message `added paths for the experiment!` followed by the MATLAB prompt `>>`. Run the commands below.
  ``` Shell
  H36MDataBase.instance;
  ```
  Set the data path to `./external/Human3.6M` and the config file directory to `./H36M_utils`.

#### Training on Human3.6M

0. Generate data.
  ```Shell
  cd $S2D3D_ROOT
  matlab -r "generate_data_h36m"
  ```

#### Running predictions on Penn Action

0. Download and extract the Penn Action dataset from `https://upenn.box.com/PennAction`.

0. Create symlinks for the downloaded Penn Action dataset. `PENN_ROOT` should contain `frames`, `labels`, and `README`.
  ```Shell
  cd $S2D3D_ROOT
  ln -s $PENN_ROOT ./external/Penn_Action
  ```

0. Prepare cropped Penn Action.
  ```Shell
  cd $S2D3D_ROOT
  matlab -r "prepare_penn_crop"
  ```

0. Generate validation set and preprocess data.
  ```Shell
  cd $S2D3D_ROOT
  matlab -r "generate_valid_penn"
  python ./tools/preprocess.py
  ```
