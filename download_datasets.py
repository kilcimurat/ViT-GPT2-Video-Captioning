from prepare_mscoco_dataset import MSCOCODataset
from prepare_vizwiz_dataset import VizWizDataset
from prepare_vatex_dataset import VatexDataset

#dt = MSCOCODataset()
#dt.download_dataset()
#dt = VizWizDataset()
#dt.download_dataset()
dt = VatexDataset()
dt.download_dataset()