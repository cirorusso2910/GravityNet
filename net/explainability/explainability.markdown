# NET.EXPLAINABILITY
**net.explainability** package

## STRUCTURE

    explainability
    |
    | - hook
    |   | backward_hook.py
    |   | forward_hook.py
    |
    | - utility
    |   | save_heatmap.py
    |   | save_image_overlay.py
    |
    | MyGradCAM.py

## DOCUMENTATION

| FOLDER      | FUNCTION              | DESCRIPTION                                                          |
|-------------|-----------------------|----------------------------------------------------------------------|
| **hook**    | backward_hook.py      | Backward hook function to capture gradients during the backward pass |
| **hook**    | forward_hook.py       | Forward hook function to capture activations during the forward pass |
| **utility** | save_heatmap.py       | Save heatmap                                                         |
| **utility** | save_image_overlay.py | Save overlay image with a Grad-CAM heatmap on top of an input image  |
|             | MyGradCAM.py          | My GradCAM implementation                                            |
