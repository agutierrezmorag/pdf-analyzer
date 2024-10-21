from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Informacion de un documento"""

    id_documento: str = Field(description="Identificador del documento")
    fuente: str = Field(description="Nombre del documento")
    empresas: list[str] = Field(
        description="Empresa o empresas a mencionadas en el documento"
    )
    autor: str = Field(description="Autor del documento")
    departamento: Optional[str] = Field(
        default=None,
        description="Departamento al cual corresponde el documento. Por ejemplo, 'Recursos Humanos', 'Ventas', 'Finanzas'",
    )
    fecha_reunion: Optional[str] = Field(
        default=None,
        description="Fecha de la reunion a la que corresponde el documento",
    )
    status: Optional[str] = Field(
        default=None,
        description="Estado del documento. Puede ser uno de los siguientes: 'pendiente', 'en proceso', 'finalizado'",
    )
    keywords: Optional[list[str]] = Field(
        default=None, description="Palabras clave del documento"
    )
    sensibilidad: Optional[str] = Field(
        description="Nivel de sensibilidad del documento. Puede ser uno de los siguientes: 'publico', 'interno', 'confidencial'"
    )
    version: Optional[str] = Field(description="Version del documento")


class ExtractionData(BaseModel):
    """Informacion extraida de un documento"""

    metadata: List[DocumentMetadata]
