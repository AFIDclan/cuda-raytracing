#pragma once

#include <iostream>
#include "MeshPrimitive.h"
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>

using namespace std;

namespace OBJLoader
{

	static MeshPrimitive load(string fp)
	{

		std::cout << "Loading OBJ file: " << fp << std::endl;


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
		vector<float2> tex_coords;

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

			if (tokens[0] == "vn")
			{
				float x = stof(tokens[1]);
				float y = stof(tokens[2]);
				float z = stof(tokens[3]);

				normals.push_back(make_float3(x, y, z));
			}

			if (tokens[0] == "vt")
			{
				float x = stof(tokens[1]);
				float y = stof(tokens[2]);

				tex_coords.push_back(make_float2(x, y));
			}
        }

        // Reset file pointer to the beginning for the second pass
        file.clear();                 // Clear any EOF or fail flags
        file.seekg(0, ios::beg);      // Move the file pointer back to the start


        vector<TrianglePrimitive> triangles;

        while (getline(file, line))
        {
            // Split the line into tokens
            istringstream iss(line);
            vector<string> tokens{ istream_iterator<string>{iss}, istream_iterator<string>{} };

            if (tokens.size() == 0)
                continue;

            // Check if the line is a face
            if (tokens[0] == "f")
            {
                // Split the face into vertex/texture/normal indices
                vector<int> vertex_indices, normal_indices, tex_indices;

                for (int i = 1; i < tokens.size(); ++i)
                {
                    size_t first_slash = tokens[i].find("/");
                    size_t second_slash = tokens[i].find("/", first_slash + 1);

                    // Extract the vertex index
                    int v_idx = stoi(tokens[i].substr(0, first_slash)) - 1;
                    vertex_indices.push_back(v_idx);

					// Extract the texture index
					if (first_slash != string::npos)
					{
						int t_idx = stoi(tokens[i].substr(first_slash + 1)) - 1;
						tex_indices.push_back(t_idx);
                    }

                    // Extract the normal index
                    if (second_slash != string::npos)
                    {
                        int n_idx = stoi(tokens[i].substr(second_slash + 1)) - 1;
                        normal_indices.push_back(n_idx);
                    }
                }

	/*			if (normal_indices.size() > 0)
                {


                    for (int i = 1; i < vertex_indices.size() - 1; i++)
                    {
						float3 norm1 = normals[normal_indices[0]];
						float3 norm2 = normals[normal_indices[i]];
						float3 norm3 = normals[normal_indices[i + 1]];

						float3 normal = normalize(norm1 + norm2 + norm3);

                        triangles.push_back(TrianglePrimitive(vertices[vertex_indices[0]], vertices[vertex_indices[i]], vertices[vertex_indices[i + 1]], normal));
                    }

                }
                else {*/
					
                    //0 (i) (i + 1)  [for i in 1..(n - 2)]

					for (int i = 1; i < vertex_indices.size() - 1; i++)
					{
                        float3 v0 = vertices[vertex_indices[i]] - vertices[vertex_indices[0]];
                        float3 v1 = vertices[vertex_indices[i + 1]] - vertices[vertex_indices[0]];
                        float3 normal = normalize(cross(v0, v1));

                        if (tex_indices.size() > 0)
                        {
							triangles.push_back(TrianglePrimitive(
								vertices[vertex_indices[0]],
								vertices[vertex_indices[i]],
								vertices[vertex_indices[i + 1]],
								normal,
								tex_coords[tex_indices[0]],
								tex_coords[tex_indices[i]],
								tex_coords[tex_indices[i + 1]]
							));



						}
						else
						{
							triangles.push_back(TrianglePrimitive(
								vertices[vertex_indices[0]],
								vertices[vertex_indices[i]],
								vertices[vertex_indices[i + 1]],
								normal
							));
						}
					}
                //}
            }
        }

        std::cout << "OBJ File: " << fp << std::endl;
		std::cout << "Loaded " << triangles.size() << " triangles" << std::endl;

		return MeshPrimitive(triangles);

	}
}
