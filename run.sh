[ -f "/home/yk/.ghcup/env" ] && source "/home/yk/.ghcup/env" # ghcup-env

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

eval "$(pyenv virtualenv-init -)" # pyenv

pyenv activate htsne
python3 setup.py build_ext --inplace
pip install .
python3 code.py
