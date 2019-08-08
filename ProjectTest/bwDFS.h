/*******************************************************************************
* File Name: AppBwLabel.c
*    Author: Showlo
*    Created: 2018-04-18
*    Abstract:
*    Reference:
*    Version:
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#pragma once
#define AFE_DATA_SIZE 809
#define CH_X_NUM             18     // ��������
#define CH_Y_NUM             32     // ��������

/* basic data type */
typedef signed char         SINT8;
typedef signed short        SINT16;
typedef signed long         SINT32;
typedef unsigned char       UINT8;
typedef unsigned short      UINT16;
typedef unsigned short      ADDR;
typedef unsigned long       UINT32;
typedef float               FP32;
typedef double              FP64;

/* ����Դ����Buff */
typedef struct {
	SINT16 ssDiffBuf[AFE_DATA_SIZE];
} ST_FrameData;

/*
*0 1 2
*3   4
*5 6 7
*/
typedef enum
{
	NEI_L_UP = 0,          // ���Ͻ�
	NEI_UP = 1,            // ��
	NEI_R_UP = 2,          // ���Ͻ�
	NEI_LEFT = 3,          // ��
	NEI_RIGHT = 4,         // ��
	NEI_L_DOWN = 5,         // ���½�
	NEI_DOWN = 6,          // ��
	NEI_R_DOWN = 7,        // ���½�
} ENUM_NEI_DIR;

/*******************************************************************************
* 3.Global constant and macro definitions using #define
*******************************************************************************/
/* ����Դ���� */
#define CH_ORDER {\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\
}
/*******************************************************************************
* 4.Static variables
*******************************************************************************/
/*******************************************************************************
* 5.Global variable or extern global variabls/functions
*******************************************************************************/
ST_FrameData g_stFrmData = { CH_ORDER };
SINT16 ssTemp[8] = { -33,-32,-31,-1,1,31,32,33 }; // ����ͨ���λ��
												/*Debug Printf*/
void AppPrintfBuf(void)
{
	int i, j, index;
	printf("\nFrameData:\n");
	printf("  ");
	for (i = 0; i < 32; i++)
	{
		printf("%2d ", i);
	}
	for (i = 0; i < CH_X_NUM; i++)
	{
		printf("\n\r%2d", i);
		for (j = 0; j < CH_Y_NUM; j++)
		{
			index = i * CH_Y_NUM + j;
			printf("%2d ", g_stFrmData.ssDiffBuf[index]);
		}
	}
	printf("\n\r");
}

/*******************************************************************************
*   Name: AppMarkNeighbours
*  Brief: ��ǵ�ǰͨ��������8����������Ƿ��������ǰ������,�ҵ���ǰ�����ݽ����еݹ���
*  Input: SINT16 *ssDiff
* Output: None
* Return: None
*******************************************************************************/
void AppMarkNeighbours(UINT16 ssChanel, UINT16 scLabel)
{
	UINT8 i;
	/* ��� */
	g_stFrmData.ssDiffBuf[ssChanel] = scLabel;
	/* 8������Ҳ��ݹ� */
	for (i = 0; i < 8; i++)
	{
		if (g_stFrmData.ssDiffBuf[ssTemp[i] + ssChanel] == 1)
		{
			AppMarkNeighbours((ssTemp[i] + ssChanel), scLabel);
		}
	}
	return;
}

void AppBwLabel(SINT16 *ssDiff)
{
	SINT8 i, j, ucTxNumVA = CH_X_NUM, ucRxNum = CH_Y_NUM;
	UINT16  usIndex = 0, ucLabel = 1; // ��ʼ1����������ǰ�������뱳������

	/* �Ծ���Դ��ͷ��β����һ�� */
	for (i = 0; i < ucTxNumVA; i++)
	{
		for (j = 0; j < ucRxNum; j++)
		{
			usIndex = i * ucRxNum + j;

			if (ssDiff[usIndex] == 1)
			{
				ucLabel++;
				AppMarkNeighbours(usIndex, ucLabel);
			}
		}
	}

	/*����ͨ�����ʼ�������Ϊ��С��Ȼ��*/
	for (i = 0; i < ucTxNumVA; i++)
	{
		for (j = 0; j < ucRxNum; j++)
		{
			usIndex = i * ucRxNum + j;

			if (ssDiff[usIndex] != 0)
			{
				ssDiff[usIndex] = ssDiff[usIndex] - 1;
			}
		}
	}
}

void test_dfs(void) {
	AppPrintfBuf();
	// ��ͨ�����㷨���
	AppBwLabel(g_stFrmData.ssDiffBuf);
	AppPrintfBuf();
}