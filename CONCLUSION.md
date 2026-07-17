### Conclusão

A Random Forest com `max_depth=5` foi escolhida por apresentar o melhor equilíbrio entre generalização e desempenho nas métricas avaliadas. O modelo atingiu recall de 75% (51/68 falhas detectadas), o que é crucial em ambientes industriais onde priorizamos reduzir falhas não detectadas.

Embora o modelo gere falsos positivos (149 alarmes falsos), esse trade-off costuma ser aceitável em manutenção preditiva, pois inspeções preventivas têm custo inferior ao de uma parada corretiva severa.

Recomendações rápidas:

- Realizar implantação piloto em ambiente controlado antes de escalar para produção.
- Ajustar o limiar de decisão ou usar calibração de probabilidades para reduzir falsos positivos, se necessário.
- Monitorar métricas (`recall`, `precision`, `f1-score`) continuamente e retreinar o modelo periodicamente com novos dados operacionais.
- Avaliar custo-benefício operacional dos alarmes para definir a política de ação (inspeção automática, notificação, etc.).

---

Arquivo gerado automaticamente a partir das conclusões em `main.ipynb`.
