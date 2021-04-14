# Visual Learning and Recognition Course Project






## Segmentation Models
Ablatation study: SegNet-LSTM utilizes the SegNet encoder to extract _scene features_ to extract scene context from images. SegNet-LSTM effectively uses the encoder portion of the SegNet architecture (i.e. VGG-16) to extract scene contexts. The authors claim that "the key is to capture encoder features
accurately rather than just increase size of the decoder."  
As such, this ablation study explores utilizing different encoder architectures to improve scene context extraction. We retain the encoder head from SegNet and swap the encoders. For this study, we compare `ResNet-18`, `ResNeXt_32x4d`, and `DenseNet-169` against the baseline `VGG-16`. We attempt to maintain a similar number of parameters for each model, as such, the number of parameteres are: 21 M, 22 M, 12 M, and 14 M respectively.