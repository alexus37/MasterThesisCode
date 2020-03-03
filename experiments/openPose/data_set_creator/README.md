# Data set creation

## steps

1. open the the MASTER_POSE.fbx in blender go to pose mode and change it as you want
2. Export as obj
3. open the obj in meshlab and remove the hands and head.
4. open obj with mts gui and create serialized version
5. run create_data_Set.ipynb
6. create masks by running. `for i in ../../data/generated/*.xml; do python openGL_save_image.py --filename $i; done`
