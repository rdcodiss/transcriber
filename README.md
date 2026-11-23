# Transcriber

Aplicativo Windows para transcrever em texto o áudio que está tocando no fone ou alto-falante. É possível escolher a saída usada pelo IVAO ou por outro aplicativo, acompanhar a atividade de áudio e exportar o texto em TXT.

## Uso

1. Baixe `Transcriber.exe` em **Releases** e execute o arquivo.
2. Em **Provider**, escolha o serviço, cole a chave e clique em **Save**.
3. Em **Output**, escolha a saída usada pelo aplicativo cujo áudio você quer transcrever.
4. Clique em **Start**. Use **Stop** para finalizar e **Save TXT** para exportar.

O Windows fornece a mistura de áudio da saída selecionada. A transcrição preserva o idioma falado. OpenAI Realtime mostra texto parcial conforme chega. Groq, Google Gemini, Deepgram, Mistral e ElevenLabs atualizam o texto por blocos. A configuração fica no `.env` ao lado do executável. O exemplo no repositório usa valores fictícios.

## Desenvolvimento no Windows

Use Python 3.11 de 64 bits. Para executar o código ou rodar os testes locais:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe transcriber.py
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Para gerar `dist\Transcriber.exe`:

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller==6.22.3
.\build.ps1 -Python .\.venv\Scripts\python.exe
```
