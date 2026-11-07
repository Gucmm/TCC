"""Schemas Pydantic — fonte única de verdade dos campos extraídos.

`InqueritoTotal` é o único contêiner canônico: toda topologia, método e
avaliador produz/consome este formato (listas diretas de pessoas).
"""
import enum
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, Field, create_model, model_validator


class ClassificacaoCrime(str, enum.Enum):
    HOMICIDIO = "Homicídio"
    LATROCINIO = "Latrocínio (roubo seguido de morte)"
    MORTE_CAUSAS_NATURAIS = "Morte por causas naturais"


class ResultadoCrime(str, enum.Enum):
    CONSUMADO = "Consumado"
    TENTADO = "Tentado"


class NaturezaAutoria(str, enum.Enum):
    CONHECIDA = "Conhecida"
    DESCONHECIDA = "Desconhecida"
    NAO_APLICAVEL = "Não aplicável"


class Pessoa(BaseModel):
    """Pessoa envolvida no inquérito."""
    nome: Optional[str] = Field(None, description="Nome da pessoa")
    cor: Optional[str] = Field(None, description="Cor da pessoa")
    sexo: Optional[str] = Field(None, description="Gênero. Infira pelo nome quando não citado")
    é_policial: bool = Field(False, description="A pessoa é policial?")
    corporacao_policial: Optional[str] = Field(None, description="Corporação (se policial)")
    policial_em_servico: Optional[bool] = Field(None, description="Estava em serviço na ocorrência?")
    profissao: Optional[str] = Field(None, description="Profissão/ocupação")
    escolaridade: Optional[str] = Field(None, description="Escolaridade")
    nacionalidade: Optional[str] = Field(None, description="Nacionalidade")
    idade: Optional[str] = Field(None, description="Idade ou faixa etária")
    antecedentes_criminais: Optional[str] = Field(
        None, description="Antecedentes criminais. Não preencha com informações contextuais (ex.: bebida)")

    @model_validator(mode="after")
    def _policial(self):
        if not self.é_policial:
            self.corporacao_policial = None
            self.policial_em_servico = None
        return self


class Vitima(Pessoa):
    armada: bool = Field(False, description="A vítima estava armada?")
    arma_da_vítima: Optional[str] = Field(None, description="Arma da vítima na ocorrência")
    faleceu: bool = Field(False, description="A vítima faleceu?")
    causa_juridica_da_morte: Optional[str] = Field(
        None, description="Causa jurídica da morte (ex.: 'homicídio por arma de fogo')")
    relacao_vitima_autor: Optional[str] = Field(None, description="Relação entre vítima e autor")

    @model_validator(mode="after")
    def _consistencia(self):
        if not self.faleceu:
            self.causa_juridica_da_morte = None
        if not self.armada:
            self.arma_da_vítima = None
        return self


class PessoasEnvolvidas(BaseModel):
    vitimas: List[str] = Field(default_factory=list, description="Nome completo de cada vítima")
    suspeitos_investigados: List[str] = Field(default_factory=list, description="Nome completo de cada suspeito/investigado")
    testemunhas: List[str] = Field(default_factory=list, description="Nome completo de cada testemunha")


class ResumoProcesso(BaseModel):
    pessoas_envolvidas: PessoasEnvolvidas = Field(description="Pessoas mencionadas e seus papéis")
    classificacao_crime: ClassificacaoCrime = Field(
        description="Classificação principal. Havendo possíveis autores, não é morte por causas naturais")


class Inquerito(BaseModel):
    resultado: Optional[ResultadoCrime] = Field(None, description="Consumado ou Tentado")
    é_feminicidio: bool = Field(False, description="Feminicídio (qualificadora do homicídio)")
    data_ocorrencia: Optional[str] = Field(None, description="Data da ocorrência")
    hora_ocorrencia: Optional[str] = Field(None, description="Hora da ocorrência")
    data_registro: Optional[str] = Field(None, description="Data do registro da ocorrência")
    delegacia_registro: Optional[str] = Field(None, description="Delegacia que registrou a ocorrência")
    tipo_de_local: Optional[str] = Field(None, description="Tipo de local (ex.: 'Residência', 'Via pública')")
    local_detalhado: Optional[str] = Field(None, description="Local específico (ex.: 'quarto', 'calçada')")
    latitude: Optional[float] = Field(None, description="Latitude do local")
    longitude: Optional[float] = Field(None, description="Longitude do local")
    municipio: Optional[str] = Field(None, description="Município da ocorrência")
    arma_utilizada: Optional[str] = Field(None, description="Arma empregada pelo autor")
    bem_roubado: Optional[str] = Field(None, description="Bem subtraído (latrocínio)")
    natureza_da_autoria: Optional[NaturezaAutoria] = Field(None, description="Conhecida ou Desconhecida")
    pericia_realizada: Optional[bool] = Field(None, description="Houve perícia?")
    prisao_em_flagrante: Optional[bool] = Field(None, description="Houve prisão em flagrante?")
    razao_arquivamento: Optional[str] = Field(None, description="Trecho literal do documento que justifica o arquivamento")


class InqueritoTotal(BaseModel):
    """Contêiner canônico."""
    resumo_processo: ResumoProcesso
    inquerito: Optional[Inquerito] = None
    vitimas: List[Vitima] = Field(default_factory=list)
    suspeitos: List[Pessoa] = Field(default_factory=list)
    testemunhas: List[Pessoa] = Field(default_factory=list)


SECTIONS = ("resumo_processo", "inquerito", "vitimas", "suspeitos", "testemunhas")
ROLES = ("vitimas", "suspeitos", "testemunhas")

_SPECS = {
    "resumo_processo": (ResumoProcesso, ...),
    "inquerito": (Optional[Inquerito], None),
    "vitimas": (List[Vitima], Field(default_factory=list)),
    "suspeitos": (List[Pessoa], Field(default_factory=list)),
    "testemunhas": (List[Pessoa], Field(default_factory=list)),
}


@lru_cache(maxsize=None)
def make_output_model(emits: tuple) -> type:
    """Modelo de saída de um nó: apenas as seções que ele emite."""
    return create_model("Out_" + "_".join(emits), **{k: _SPECS[k] for k in emits})
