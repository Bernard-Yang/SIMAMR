# SIMAMR
# MedDiaAMR



Code for the paper "[Structured Information Matters: Incorporating Abstract Meaning Representation into LLMs for Improved Open-Domain Dialogue Evaluation](https://arxiv.org/pdf/2404.01129)" 

### Dataset 

To get the AMR graph (.concept and .path files) of each sentence, you can follow the preprocess in [https://github.com/Soistesimmer/AMR-multiview](https://github.com/Soistesimmer/AMR-multiview)

The structure of dataset should be like this:
```
├── dataset
       └── `new.src`
       └── `new.tgt`
       └── `new.path`
       └── `new.concept`
       └── `con.concept`
       └── `con.path` 
```
where src is the context and tgt is the response, concept and path are the AMR graph files.
### Preprocessing 
```
python preprocess_context.py
```
### Training 
```
python run-context.py
```

### Citation
If you found this repository or paper is helpful to you, please cite our paper.
```
@article{yang2024structured,
  title={Structured Information Matters: Incorporating Abstract Meaning Representation into LLMs for Improved Open-Domain Dialogue Evaluation},
  author={Yang, Bohao and Zhao, Kun and Tang, Chen and Zhan, Liang and Lin, Chenghua},
  journal={arXiv preprint arXiv:2404.01129},
  year={2024}
}
```
