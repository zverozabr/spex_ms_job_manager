import glob
import json
import importlib.util
import importlib


def start_scenario(script="", part="", **kwargs):
    print(script, part)
    manifest_arr = glob.glob(f'{script}/{part}.json', recursive=True)
    for file in manifest_arr:
        data = json.load(open(file))
        if depends := data.get('depends_script'):
            for item in depends:
                res = start_scenario(script=script, part=item, **kwargs)
                kwargs.update(res)
        module = importlib.import_module(f'.{data["script_path"]}', package=script)
        res = module.run(**kwargs)
        kwargs.update(res)
        return kwargs


# result = start_scenario(script='segmentation', part='denoise', image_path='2.ome.tiff', channel_list=[0, 2, 3])
#  1, 0.5, 1, 98.5
# result = start_scenario(
#     script='segmentation',
#     part='stardist_cellseg',
#     image_path='2.ome.tiff',
#     kernal=5,
#     channel_list=[0, 2, 3],
#     scaling=1,
#     threshold=0.5,
#     _min=1,
#     _max=98.5
# )

# result = start_scenario(
#          script='segmentation',
#          part='deepcell_segmentation',
#          image_path='2.ome.tiff',
#          channel_list=[0, 2, 3],
#          kernal=5,
#          mpp=0.39)

# result = start_scenario(
#     script='segmentation',
#     part='rescues_cells',
#     image_path='2.ome.tiff',
#     channel_list=[0, 2, 3],
#     kernal=5,
#     mpp=0.39)


result = start_scenario(
    script='segmentation',
    part='feature_extraction',
    image_path='2.ome.tiff',
    channel_list=[0, 2, 3],
    kernal=5,
    mpp=0.39,
    dist=8)

print(result)

