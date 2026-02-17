Este diretório é reservado para **assets oficiais** da métrica do Kaggle (não comitar no git).

Coloque aqui:
- `metric.py` (script oficial de avaliação do Kaggle)
- o binário `USalign` correspondente (se o `metric.py` exigir execução via subprocess)

Recomendação:
- mantenha internet **desativada** no notebook Kaggle de submissão;
- para validação local, use o comando `python -m rna3d_local score-local-kaggle-official ...`.

Contratos:
- O `metric.py` deve expor uma função `score(sol_df, sub_df, row_id_column_name='ID') -> float`.
