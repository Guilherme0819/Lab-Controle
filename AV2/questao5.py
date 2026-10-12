"""
Resolução da Questão 5: Adaptação do Galvanômetro
Cálculo dos resistores para Amperímetro e Voltímetro.
"""

def dimensionar_amperimetro(i_total_desejada, i_g_max=1.5, r_g=2.5):
    """Calcula o resistor Shunt (em paralelo) para o Amperímetro."""
    if i_total_desejada <= i_g_max:
        return "Erro: A corrente desejada deve ser maior que 1.5A"
    
    corrente_shunt = i_total_desejada - i_g_max
    r_shunt = (r_g * i_g_max) / corrente_shunt
    return r_shunt

def dimensionar_voltimetro(v_total_desejada, i_g_max=1.5, r_g=2.5):
    """Calcula o resistor Multiplicador (em série) para o Voltímetro."""
    v_max_galvanometro = r_g * i_g_max
    if v_total_desejada <= v_max_galvanometro:
        return f"Erro: A tensão deve ser maior que {v_max_galvanometro}V"
    
    r_multiplicador = (v_total_desejada / i_g_max) - r_g
    return r_multiplicador

if __name__ == "__main__":
    print("--- Adaptação do Galvanômetro (Questão 5) ---")
    
    # a) Adaptação para Amperímetro (Exemplo: medir até 10A)
    alvo_corrente = 10.0 
    r_shunt = dimensionar_amperimetro(alvo_corrente)
    print(f"\na) Para transformar num Amperímetro de {alvo_corrente}A:")
    print(f" -> Ligar um Resistor Shunt (em PARALELO) de: {r_shunt:.4f} Ohms")

    # b) Adaptação para Voltímetro (Exemplo: medir até 120V)
    alvo_tensao = 120.0 
    r_mult = dimensionar_voltimetro(alvo_tensao)
    print(f"\nb) Para transformar num Voltímetro de {alvo_tensao}V:")
    print(f" -> Ligar um Resistor Multiplicador (em SÉRIE) de: {r_mult:.4f} Ohms")