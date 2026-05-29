<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="1.0" 
                xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:xs="http://www.w3.org/2001/XMLSchema">
    <xsl:output method="text" encoding="UTF-8"/>
    <xsl:include href="warnings.xsl"/>

    <xsl:template match="/xs:schema">
        <xsl:call-template name="perl-header"/>
        <xsl:apply-templates select="xs:element[@type='MoveRecord']"/>
        <xsl:text>&#10;1;&#10;</xsl:text>
    </xsl:template>

    <xsl:template match="xs:element[@type='MoveRecord']"> 
        <xsl:text>my %move_record = map { lc($_) => "" } qw(</xsl:text>
        <xsl:apply-templates select="//xs:complexType[@name=current()/@type]"/>
        <xsl:text>);&#10;</xsl:text>
    </xsl:template>

    <xsl:template match="xs:complexType">
        <xsl:apply-templates select="xs:sequence/xs:element"/>
    </xsl:template>

    <xsl:template match="xs:sequence/xs:element">
        <xsl:value-of select="@name"/><xsl:text> </xsl:text>
    </xsl:template>

    <xsl:template name="perl-header">
        <xsl:text># </xsl:text>
        <xsl:call-template name="warning-header"/>
        <xsl:text>package move_record;&#10;&#10;</xsl:text>
        <xsl:text>use Exporter qw(import);&#10;</xsl:text>
        <xsl:text>our @EXPORT_OK = qw(%move_record);&#10;&#10;</xsl:text>
    </xsl:template>

</xsl:stylesheet>
