
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image

def matplotlib_from_base64(encoded:str,title:str=None,figsize :tuple=(8,6)):
    imag_data= base64.b64decode(encoded)
    buf=BytesIO(imag_data)
    img=Image.open(buf)
    fig,ax =plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis('off')
    if title:
        ax.set_title(title)
    plt.show()
    return fig,ax
    
    