# Installation
install the python requirements:
pip install -r requirements.txt
Install ollama framework on your system: https://ollama.com/

# Train model 
run the script train.py in the package JointModel,
configure the parameters in the file parameters.py accordingly
# Run webservice for evaluation
download the content from this ftp folder: https://files.dice-research.org/projects/EL-Context_Augmentation/

run the script webservice.py in the folder experiments. 
Configure the parameters accordingly espacially the parameter for the according ollama model (--llama_model)
for detailed parameters see the file parameters.py in the same directory
