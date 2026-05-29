<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" 
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xs="http://www.w3.org/2001/XMLSchema">
    <xsl:output method="text" encoding="UTF-8"/>

    <xsl:include href="warnings.xsl"/>

    <xsl:template match="/xs:schema">
        <xsl:call-template name="c-header"/>
        <xsl:apply-templates select="xs:element[@type='MoveRecord']"/>
        <xsl:call-template name="c-footer"/>
    </xsl:template>

    <xsl:template match="xs:element[@type='MoveRecord']">
        <xsl:apply-templates select="//xs:complexType[@name=current()/@type]"/>
    </xsl:template>

    <xsl:template match="xs:complexType[@name='MoveRecord']">
        <xsl:variable name="fieldType" select="xs:sequence
                                               /xs:element
                                               /@type"/>                                      
        <!-- First define structs for complex types -->
        <xsl:apply-templates select="//xs:complexType[@name=$fieldType]"/>

        <!-- Then build the main move_record struct -->
        <xsl:text>typedef struct {&#10;</xsl:text>
        <xsl:apply-templates select="xs:sequence/xs:element"/>
        <xsl:text>} Move_Record;&#10;</xsl:text>
    </xsl:template>

    <!-- Defines struct structure -->
    <xsl:template match="xs:complexType">
        <xsl:param name="fieldName"/>

        <xsl:text>typedef struct {&#10;</xsl:text>
        <xsl:apply-templates select="xs:sequence/xs:element"/>
        <xsl:text>} </xsl:text>
        <xsl:value-of select="@name"/>
        <xsl:text>;&#10;&#10;</xsl:text>
    </xsl:template>

    <!-- \tchar $(name)[$(length)];\n -->
    <xsl:template match="xs:simpleType">
        <xsl:param name="fieldName" select="default"/>
        <xsl:variable name="length" select="xs:restriction/xs:length/@value"/>

        <xsl:text>&#9;char </xsl:text>
        <xsl:value-of select="$fieldName"/>

        <xsl:if test="$length &gt; 1">
            <xsl:text>[</xsl:text>
            <xsl:value-of select="$length"/>
            <xsl:text>]</xsl:text>
        </xsl:if>

        <xsl:text>;&#10;</xsl:text>
    </xsl:template>

    <xsl:template match="xs:element">
        <xsl:variable name="fieldName" select="@name"/>
        <xsl:variable name="simpleType" 
                      select="//xs:simpleType[@name=current()/@type]"/>
        <xsl:variable name="complexType" 
                      select="//xs:complexType[@name=current()/@type]"/>

        <xsl:choose>
            <xsl:when test="$simpleType">
                <xsl:apply-templates select="$simpleType">
                    <xsl:with-param name="fieldName" select="$fieldName"/>
                </xsl:apply-templates>
            </xsl:when>

            <!-- \t$(complexType) $(name);\n -->
            <xsl:when test="$complexType">
                <xsl:text>&#9;</xsl:text>
                <xsl:value-of select="@type"/>
                <xsl:text> </xsl:text>
                <xsl:value-of select="@name"/>
                <xsl:text>;&#10;</xsl:text>
            </xsl:when>

            <xsl:otherwise/>
        </xsl:choose>
    </xsl:template>

    <xsl:template name="c-header">
        <text>// </text>
        <xsl:call-template name="warning-header"/>

        <text>#ifndef MOVE_RECORD_H&#10;</text>
        <text>#define MOVE_RECORD_H&#10;&#10;</text>
    </xsl:template>

    <xsl:template name="c-footer">
        <xsl:text>&#10;#endif /* MOVE_RECORD_H */&#10;</xsl:text>
    </xsl:template>

</xsl:stylesheet>
