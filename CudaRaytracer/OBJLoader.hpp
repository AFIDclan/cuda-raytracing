#pragma once

#include <iostream>
#include "MeshPrimitive.h"
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

namespace OBJLoader
{
	static MeshPrimitive load(string fp)
	{


		// Open the file
		ifstream file(fp);
		if (!file.is_open())
		{
			cout << "Could not open file " << fp << endl;
			exit(1);
		}

		// Read the file line by line
		string line;

		vector<float3> vertices;
		vector<float3> normals;
		vector<int3> faces;


		while (getline(file, line))
		{
			// Split the line into tokens
			istringstream iss(line);
			vector<string> tokens{ istream_iterator<string>{iss}, istream_iterator<string>{} };

			if (tokens.size() == 0)
				continue;
			
			// Check if the line is a vertex
			if (tokens[0] == "v")
			{
				float x = stof(tokens[1]);
				float y = stof(tokens[2]);
				float z = stof(tokens[3]);

				vertices.push_back(make_float3(x, y, z));
			}

			if (tokens[0] == "f")
			{
				int v1 = stoi(tokens[1]) - 1;
				int v2 = stoi(tokens[2]) - 1;
				int v3 = stoi(tokens[3]) - 1;

				faces.push_back(make_int3(v1, v2, v3));
			}
		}

		vector<TrianglePrimitive> triangles;

		for (int i = 0; i < faces.size(); i++)
		{
			int3 face = faces[i];


			float3 a = vertices[face.x];
			float3 b = vertices[face.y];
			float3 c = vertices[face.z];

			triangles.push_back(TrianglePrimitive(a, b, c, make_float3(0.0f, 0.5f, 1.0f)));
		}

		return MeshPrimitive(triangles);

	}
}
