import spex_segment as sp
import skimage.segmentation as segmentation
import numpy as np
from tifffile import TiffWriter


files = ['1.tif.ome.tif']

for image in files:
    # Load Image
    Image, channel = sp.load_tiff(image, is_mibi=True)
    (c, y, x) = Image.shape
    # Denoise Image
    # list = ['dsDNA', 'H3K9ac', 'H3K27me3']
    # list = ['131Xe', '138Ba', 'PDPN']
    list = [None, None, None]

    to_denoise = []
    for i in range(0, len(list), 1):
        # try:
        index = channel.index(list[i])
        # except ValueError:
        #     continue
        to_denoise.append(index)
    to_denoise.sort()

    median_Image = sp.median_denoise(Image, 5, to_denoise)

    # Run Segmentation
    # list = ['dsDNA', 'H3K9ac', 'H3K27me3']
    list = []

    to_merge = []
    for i in range(0, len(list), 1):
        try:
            index = channel.index(list[i])
            to_merge.append(index)
        except ValueError:
            continue
    to_merge.sort()

    stardist_label = sp.stardist_cellseg(median_Image, to_merge, 1, 0.5, 1, 98.5)
    # cellpose_label=sp.cellpose_cellseg(median_Image, index,12, 1)
    deepcell_label = sp.deepcell_segmentation(Image, to_merge, 0.39)

    # index=channel.index('H3K27me3')
    new_label = sp.rescue_cells(Image, to_merge, stardist_label)
    # new_label2=sp.rescue_cells(Image,index, cellpose_label)
    # new_label3=sp.rescue_cells(Image,index, deepcell_label)

    # Dilate Cells
    expanded_label = sp.simulate_cell(new_label, 10)
    # expanded_label2=sp.simulate_cell(new_label2, 10)
    expanded_label3 = sp.simulate_cell(deepcell_label, 8)

    # Extract Features
    df = sp.feature_extraction(Image, expanded_label, channel)
    df2 = sp.feature_extraction(Image, expanded_label3, channel)

    # Save Feature Data
    csvname = image.split(".tiff")[0]+'_stardist.csv'
    df.to_csv(csvname, index=False)
    csvname = image.split(".tiff")[0]+'_deepcell.csv'
    df2.to_csv(csvname, index=False)

    # Save Image of segmentation
    imagename = image.split(".tiff")[0]+'_label.ome.tiff'

    contour = segmentation.find_boundaries(expanded_label, connectivity=1, mode='thick', background=0)
    # contour2=segmentation.find_boundaries(expanded_label2, connectivity=1, mode='thick', background=0)
    contour3 = segmentation.find_boundaries(expanded_label3, connectivity=1, mode='thick', background=0)
    # list = ['131Xe', '138Ba', 'PDPN']
    # pseudoIF=np.stack((Image[channel.index('dsDNA')],Image[channel.index('H3K9ac')],
    # Image[channel.index('H3K27me3')],contour,contour3), axis=0)
    pseudoIF = np.stack((Image[0], contour, contour3), axis=0)

    with TiffWriter(imagename, bigtiff=True) as tif:
        options = dict(tile=(512, 512), photometric='minisblack')
        # tif.write(pseudoIF, **options, metadata={'PhysicalSizeX':0.39,'PhysicalSizeY':0.39,
        # 'Channel': {'Name': ["dsDNA","H3K9ac","H3K27me3", "Stardist","DeepCell"]}})
        tif.write(pseudoIF, **options, metadata={'PhysicalSizeX': 0.39, 'PhysicalSizeY': 0.39,
                                                 'Channel': {'Name': ["131Xe", "138Ba", "PDPN", "Stardist", "DeepCell"]}}
                  )


