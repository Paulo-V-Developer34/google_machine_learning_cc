# Google Machine Learning Crash Course (Google MLCC)

Essas são as atividades abordadas no curso, porém feitas de forma um pouco diferente. Vale ressaltar que foi utilizado o UV python para gerenciar e executar o projeto.

## UV python

UV python é uma ferramenta que faz muito além de apenas gerenciar dependências python, ele também pode executar projetos python, ambientes virtuais, "tools" e muito mais

## Iniciar projeto

1. tenha o UV instalado, para isso execute ```pip install uv```
2. selecione o interpretador python do venv do UV (caso não apareça execute o comando do tópico 2, copie o caminho da pasta venv que for criada e adicione-a aos caminhos de ambiente virtual)
3. execute ```uv run ./linear_regression/main.py``` para rodar o projeto

## Problemas

1. Caso os modulos python não sejam reconhecidos verifique o interpretador python
2. Caso o passo anterior não funcione execute ```uv tree``` para ver as dependências, caso as dependências necessárias não apareçam execute ```uv install```
3. Caso o erro ```ImportError: DLL load failed while importing _ml_dtypes_ext: O nome do arquivo ou a extensão é muito grande.``` apareça você deve mudar o projeto para um caminho mais curto para seu Sistema Operacional
4. Caso o projeto não reconheça módulos criados por você verifique a pasta "__pycache__" gerada, caso o seu arquivo não apareça lá verifique a importação feita a ela
5. Caso a importação de outro arquivo python (módulo) falhe lembre-se de sempre usar a pasta do projeto como o root dos caminhos dos seus arquivos.
