module purge
module load python/3.9.6

echo "loading module done"

echo "Creating new virtualenv"

virtualenv ~/$1
source ~/$1/bin/activate

echo "Activating virtual env"


# pip install --no-index --upgrade pip

pip install -r requirements.txt
