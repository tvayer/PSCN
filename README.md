PSCN is a python3 implementation of the paper "Learning Convolutional Neural Networks for Graphs" by Mathias Niepert, Mohamed Ahmed and Konstantin Kutzkov (https://arxiv.org/abs/1605.05273)

- Requires : networkx >=2, keras, pynauty

- To install pynauty :
	- Go to folder pynauty-0.6.0
	- Build pynauty : "make pynauty"
	- Set user : "make user-ins pynauty"
	- Install : "pip install ."

Questions about the paper : 

- There is no indication about the labeling procedure used for the classification : is it chosen among a bunch of procedures (which one ?) before the classification using Theorem 2, or is it fixed to one procedure (eigenvector centrality, degree, ...). In this implementation it is fixed to be the betweeness centrality.

- The convolution seems to be made over the dicrete labels/attributes of the graph nodes in the classification (molecules represented by 0,1,2 etc for MUTAG, PTC..) : what is the sense of such a convolution ?

- For "dummy nodes" (when the size of the receptive field is higher than the size of the graph) which node attribute should be used ??




