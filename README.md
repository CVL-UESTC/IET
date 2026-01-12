<p align="center">
  <h1 align="center">From Local Windows to Adaptive Candidates via Individualized Exploratory: Rethinking Attention for Image Super-Resolution</h1>
  <p align="center">
    <a href="https://openreview.net/profile?id=%7EChunyu_Meng5">Chunyu Meng</a>
    ·
    <a href="https://scholar.google.com/citations?user=CsVTBJoAAAAJ&hl=zh-CN">Wei Long</a>
    ·
    <a href="https://scholar.google.com/citations?user=-kSTt40AAAAJ&hl=zh-CN">Shuhang Gu</a>
  </p>

[//]: # (  <h3 align="center">CVPR 2025</h3>)

[//]: # (  <h3 align="center">)
[//]: # (  </h3>)
</p>

## Aabstract

<p align="left">
Single Image Super-Resolution is a classic computer vision problem that aims to reconstruct a high-resolution (HR) image from a low-resolution (LR) input. Transformer-based methods have achieved remarkable results in such tasks because they can capture non-local dependencies in low-quality input images. However, their feature-intensive modeling leads to high computational complexity. To reduce the cost, most existing approaches divide images into groups and restrict attention computation within each group, which inevitably limits flexibility and hinders each token from identifying its most relevant counterparts. To address this limitation, we propose the Individualized Exploratory Transformer (IET), featuring a novel Individualized Exploratory Attention (IEA) mechanism that allows each token to form content-aware and independent attention candidates. This token-adaptive design enables more precise and efficient information enhancement across the network. Extensive experiments on multiple standard SR benchmarks demonstrate the effectiveness of our approach, achieving state-of-the-art performance under comparable computational complexity. 
</p>


<p align="center">
  <a href="">
    <img src="figures/IDESplat_m.png" alt="the Overview Architecture of Individualized Exploratory Attention (IEA)" width="100%">
  </a>
</p>


The complete training and inference code, along with the pretrained model for IET, will be released soon.
