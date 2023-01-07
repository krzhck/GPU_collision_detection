#pragma once
#include<GL/glut.h>


class Light
{
public:
	GLfloat Color[4] = { 0, 0, 0, 0 }; //������ɫ
	GLfloat Ambient[4] = { 0, 0, 0, 0 }; //������
	GLfloat Diffuse[4] = { 0, 0, 0, 0 }; //������
	GLfloat Specular[4] = { 0, 0, 0, 0 }; //���淴��
	GLfloat Position[4] = { 0, 0, 0, 0 }; //����ָ��

	Light() {}

	//��ʼ��
	void Init(GLfloat color[], GLfloat ambient[], GLfloat diffuse[], GLfloat specular[], GLfloat position[])
	{
		for (int i = 0; i < 3; i++)
		{
			Color[i] = color[i];
			Ambient[i] = ambient[i];
			Diffuse[i] = diffuse[i];
			Specular[i] = specular[i];
			Position[i] = position[i];
		}
		//͸���ȣ�1
		Color[3] = 1.0;
		Ambient[3] = 1.0;
		Diffuse[3] = 1.0;
		Specular[3] = 1.0;

		//����Զ��ƽ�й�
		Position[3] = 1.0; 
	}
};