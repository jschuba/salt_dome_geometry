import datetime
import os


def gen_png_filename(stub: str, folder='data', subfolder=datetime.date.today().strftime('%Y%m%d'), extension='png'):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = os.path.join(folder, subfolder)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'{stub}_{now}.{extension}')
