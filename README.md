# SIMAMR

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


