#include<GL/glut.h>
#include<iostream>
#include<math.h>
#include<windows.h>
#include "coord.hpp"
#include "ball.hpp"
#include "light.hpp"
#include "camera.hpp"
#include "wall.hpp"
#include "ballset.hpp"
#include "collision.cuh"
using namespace std;


//全局常量
const int WindowSizeX = 800, WindowSizeY = 600, WindowPosX = 100, WindowPosY = 100;
const char WindowName[] = "collision detection";
const float TimeOnce = 0.02; //刷新时间
const int BallNum = 5;
const float XRange = 10, ZRange = 10, Height = 20, MaxRadius = 1; //场景的X,Y,Z范围（-X,X),(0,H),(-Z,Z)

//光照，相机
Camera TheCamera(20.0f, 10.0f);
Light TheLight;

//物体
Wall Walls[6]; //边界

BallSet Balls(XRange, Height, ZRange, BallNum, MaxRadius, TimeOnce);


//初始化函数集合
//初始化窗口
void InitWindow()
{
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(WindowSizeX, WindowSizeY);
	glutInitWindowPosition(WindowPosX, WindowPosY);
	glutCreateWindow(WindowName);
	const GLubyte* OpenGLVersion = glGetString(GL_VERSION);
	const GLubyte* gluVersion = gluGetString(GLU_VERSION);
	printf("OpenGL实现的版本号：%s\n", OpenGLVersion);
	printf("OGLU工具库版本：%s\n", gluVersion);
	int dev = 0;
	cudaDeviceProp devProp;
	if (cudaGetDeviceProperties(&devProp, dev) == cudaSuccess)
	{
		std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
		std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
		std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
		std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
		std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
		std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
	}
}

//初始化光照
void InitLight()
{
	//设置着色模式
	glShadeModel(GL_SMOOTH);
	//设置初始背景色，清除颜色缓存和深度缓存
	glClearColor(TheLight.Color[0], TheLight.Color[1], TheLight.Color[2], TheLight.Color[3]);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//设置光源信息
	glLightfv(GL_LIGHT0, GL_AMBIENT, TheLight.Ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, TheLight.Diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, TheLight.Specular);
	glLightfv(GL_LIGHT0, GL_POSITION, TheLight.Position);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	//设置深度检测，即只绘制最前面的一层
	glEnable(GL_DEPTH_TEST);
}

//初始化边界和地板
void InitWalls()
{
	//8个点
	Coord DownA(-XRange, 0, -ZRange);
	Coord DownB(-XRange, 0, ZRange);
	Coord DownC(XRange, 0, -ZRange);
	Coord DownD(XRange, 0, ZRange);
	Coord UpA(-XRange, Height, -ZRange);
	Coord UpB(-XRange, Height, ZRange);
	Coord UpC(XRange, Height, -ZRange);
	Coord UpD(XRange, Height, ZRange);

	//设置地板和挡板位置
	Walls[0].InitPos(DownA, DownB, DownD, DownC); // bottom
	Walls[1].InitPos(DownA, DownB, UpB, UpA);
	Walls[3].InitPos(DownC, DownD, UpD, UpC);
	Walls[2].InitPos(DownA, DownC, UpC, UpA);
	Walls[4].InitPos(DownB, DownD, UpD, UpB);
	Walls[5].InitPos(UpA, UpB, UpD, UpC); // top

	GLfloat color_bottom[4] = { 1.0, 1.0, 1.0 , 1};
	GLfloat ambient_bottom[4] = { 0.4, 0.4, 0.4 , 1};
	GLfloat diffuse_bottom[4] = { 0.4, 0.4, 0.4 , 1};
	GLfloat specular_bottom[4] = { 0.2, 0.2, 0.2 , 1};
	GLfloat shininess_bottom = 20;

	Shader shader_bottom(color_bottom, ambient_bottom, diffuse_bottom, specular_bottom, shininess_bottom);
	Walls[0].InitColor(shader_bottom);

	//设置四周挡板材质
	GLfloat color_border[3] = { 1.0, 1.0, 1.0 };
	GLfloat ambient_border[3] = { 0.2, 0.2, 0.2 };
	GLfloat diffuse_border[3] = { 0.2, 0.2, 0.2 };
	GLfloat specular_border[3] = { 0.2, 0.2, 0.2 };
	GLfloat shininess_border = 40;
	Shader shader_border(color_border, ambient_border, diffuse_border, specular_border, shininess_border);
	for (int i = 1; i < 5; i++)
	{
		Walls[i].InitColor(shader_border);
	}
}




//初始化的主函数
void InitScene()
{

	InitLight();
	InitWalls();
	Balls.InitBalls();
}

//绘制函数集合
//设置相机位置
void SetCamera()
{
	glLoadIdentity();
	Coord camera_Pos = TheCamera.Pos;//这就是视点的坐标  
	Coord camera_center = TheCamera.LookCenter;//这是视点中心坐标
	gluLookAt(camera_Pos.x, camera_Pos.y, camera_Pos.z, camera_center.x, camera_center.y, camera_center.z, 0, 1, 0); //从视点看远点,y轴方向(0,1,0)是上方向  
}

//绘制2边界和地板
void RenderWalls()
{
	for (int i = 0; i < 3; i++)
	{
		glColor3f(Walls[i].WallShader.Color[0], Walls[i].WallShader.Color[1], Walls[i].WallShader.Color[2]);
		glMaterialfv(GL_FRONT, GL_AMBIENT, Walls[i].WallShader.Ambient);
		glMaterialfv(GL_FRONT, GL_DIFFUSE, Walls[i].WallShader.Diffuse);
		glMaterialfv(GL_FRONT, GL_SPECULAR, Walls[i].WallShader.Specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, Walls[i].WallShader.Shininess);


		glBegin(GL_POLYGON);
		glVertex3f(Walls[i].Vertexes[0].x, Walls[i].Vertexes[0].y, Walls[i].Vertexes[0].z);
		glVertex3f(Walls[i].Vertexes[1].x, Walls[i].Vertexes[1].y, Walls[i].Vertexes[1].z);
		glVertex3f(Walls[i].Vertexes[2].x, Walls[i].Vertexes[2].y, Walls[i].Vertexes[2].z);
		glVertex3f(Walls[i].Vertexes[3].x, Walls[i].Vertexes[3].y, Walls[i].Vertexes[3].z);
		glEnd();
		glFlush();
	}

}


//绘制的主函数
void RenderScene()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	//清除颜色缓存
	SetCamera();//设置相机
	RenderWalls();//绘制地板和边框
	Balls.UpdateBalls();
	Balls.RenderBalls();//更新和绘制小球
	glutSwapBuffers();
}

//全局定时器
void OnTimer(int value)
{
	glutPostRedisplay();//标记当前窗口需要重新绘制，调用myDisplay()
	glutTimerFunc(20, OnTimer, 1);
}


//交互函数集合
//处理鼠标点击 
void OnMouseClick(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		TheCamera.MouseDown(x, y);
	}
}

//处理鼠标拖动  
void OnMouseMove(int x, int y)
{
	TheCamera.MouseMove(x, y);
}

//处理键盘点击（WASD）
void OnKeyClick(unsigned char key, int x, int y)
{
	int type = -1;
	if (key == 'w')
	{
		type = 0;
	}
	else if (key == 'a')
	{
		type = 1;
	}
	else if (key == 's')
	{
		type = 2;
	}
	else if (key == 'd')
	{
		type = 3;
	}
	TheCamera.KeyboardMove(type);
}

//处理键盘点击（前后左右）
void OnSpecialKeyClick(GLint key, GLint x, GLint y)
{
	int type = -1;
	if (key == GLUT_KEY_UP)
	{
		type = 0;
	}
	if (key == GLUT_KEY_LEFT)
	{
		type = 1;
	}
	if (key == GLUT_KEY_DOWN)
	{
		type = 2;
	}
	if (key == GLUT_KEY_RIGHT)
	{
		type = 3;
	}
	TheCamera.KeyboardMove(type);
}


//reshape函数
void reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(75.0f, (float)w / h, 1.0f, 1000.0f);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	InitWindow();             //初始化窗口
	InitScene();              //初始化场景
	glutReshapeFunc(reshape); //绑定reshape函数
	glutDisplayFunc(RenderScene); //绑定显示函数
	glutTimerFunc(20, OnTimer, 1);  //启动计时器
	glutMouseFunc(OnMouseClick); //绑定鼠标点击函数
	glutMotionFunc(OnMouseMove); //绑定鼠标移动函数
	glutKeyboardFunc(OnKeyClick);//绑定键盘点击函数
	glutSpecialFunc(OnSpecialKeyClick);//绑定特殊键盘点击函数
	glutMainLoop();
}
