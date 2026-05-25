<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0" 
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xs="http://www.w3.org/2001/XMLSchema">

    <xsl:key name="simpleType"  match="xs:simpleType" use="@name"/>
    <xsl:key name="complexType" match="xs:complexType" use="@name"/>

<!-- ========================= Define Indentations ========================= -->
    <xsl:variable name="root-indent"    select="'       '"/>
    <xsl:variable name="std-indent"     select="'  '"/>
    
    <xsl:output method="text" encoding="UTF-8"/>

    <xsl:template match="xs:schema/xs:element">
        <xsl:value-of select="$root-indent"/>
        <xsl:text>01 MOVE-RECORD.&#10;</xsl:text>
        <xsl:apply-templates select="xs:element"/>
    </xsl:template>

    <xsl:template match="xs:element">
        <xsl:value-of select="$root-indent"/>

        <xsl:param name="level"/>
        <xsl:variable name="leafName" select="key('simpleType', @type)"/>
        <xsl:variable name="nodeName" select="key('complexType', @type)"/>

        <xsl:choose>
            <xsl:when test="$leafName">
                <xsl:value-of select="$level"></xsl:value-of>
    
            </xsl:when>

            <xsl:when test="$nodeName">
                <xsl:with-param name="level" select="$level + 5"/>
            </xsl:when>
        </xsl:choose>
    </xsl:template>

</xsl:stylesheet>
