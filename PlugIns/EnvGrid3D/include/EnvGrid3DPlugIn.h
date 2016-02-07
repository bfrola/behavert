// ----------------------------------------------------------------------------
// This source file is part of BehaveRT 
// http://isis.dia.unisa.it/projects/behavert/
//
// Copyright (c) 2008-2010 ISISLab - University of Salerno
// Original author: Bernardino Frola <frola@dia.unisa.it>
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

// ----------------
// Change log
//
// 01-09 bf: Created
//
// ----------------
#pragma once

#include "BehaveRT.h"
#include "envgrid3d_kernel.cuh"
#include "radixsort.cuh"

// ----------------

/// EnvGrid3D plugIn namespace
namespace EnvGrid3D
{
	int3 calcGridPosH(float4 p, float3 worldOrigin, float3 cellSize);
	BehaveRT::uint calcGridHashH (int3 gridPos, uint3 gridSize);

	typedef std::vector<char*> SupportedDataTypes;

	typedef std::pair<InstallableEntity*, const type_info*> FeatureToReorder;
	typedef std::vector<FeatureToReorder> FeaturesToReorder;

	/**
		\brief This plugIn provides a spactial DB based on m_CommonRes's grid.

		When an other plugIn extends this one, it inheritates the features contained in EnvGrid3DFields and parameters in EnvGrid3DParams
	*/
	template <class Super>
	class EnvGrid3DPlugIn: public Super, public SimEnginePlugIn
	{
	public:
		// /////////////////////////////////////////////////////
		// Constructor/Descructor

		/// XXX install/unistall plugin shoulde be automatic
		EnvGrid3DPlugIn() { SimEnginePlugIn::installPlugIn(); }
		~EnvGrid3DPlugIn() { SimEnginePlugIn::uninstallPlugIn(); }
		
		const std::string name() { return "EnvGrid3DPlugIn"; }	

		const DependenciesList plugInDependencies() 
		{ 
			DependenciesList dependencies;
			dependencies.push_back("Body3DPlugIn");
			return dependencies;	
		}

		// ////////////////////////////////////////////////////
		// Methods
	private:
		/// @override
		void install();
		
		/// @override
		void uninstall();

	public:
		/// @override
		void reset();

		/// @override
		void update(const float elapsedTime);

		// Custom operations
	public:
		
		/**	
			When a feature is included to the list FeatureToReorder, at each frame, 
			the it will be reordered based on the hash scheme
			\param entity feature to reorder, such as, a DeviceArrayWrapper
			\param type the type_info of the feature
		*/
		void addToFeaturesToReorder(InstallableEntity* entity, const type_info* type);

		/// Reorder the feature added to the specific list \see addToFeaturesToReorder
		void reorderSimData();

		/// Returns the position feature
		BehaveRT::DeviceArrayWrapper<float4>* getPos() {return m_Pos; }
		BehaveRT::DeviceArrayWrapper<uint2>* getHash() {return m_AgentHash; }

		// Return the sorted index
		BehaveRT::uint getSortedIndex(BehaveRT::uint index, bool copyFromDevice = true ) { 
			return m_AgentHash->getHostArrayElement( index, copyFromDevice ).y; }

		// LINKING PROBLEMS IF USED OUT OF THIS FILE
		/// Calculate worldCenter based on worldOrigin and worldRadius
		float3 getWorldCenter();

	protected:
		// LINKING PROBLEMS IF USED OUT OF THIS FILE
		/// Setup worldOrigin based on worldCenter and worldRadius
		void setWorldCenter( float3 worldCenter );


	private:
		/// Initializes the parameters regarding the world size
		void setupWorldRadius();

		/// Initializes the grid
		void setupGrid();
		
		float3 getAugmentedRadius( float3 worldRadius, float3 cellSize );
		

		/// Calculates automatically how many bodies at most can enter into a cell. 
		/// Used the body common radius \see Body3DParams and the cell size. \see EnvGrid3DParams
		void setupMaxBodiesPerCell();

		// ////////////////////////////////////////////////////
		// Fields	
	protected:
		BehaveRT::DeviceArrayWrapper<float4>* m_Pos;
		BehaveRT::DeviceArrayWrapper<uint2>* m_AgentHash;

		FeaturesToReorder m_FeaturesToReorder;
	};
}

using namespace EnvGrid3D;

// --------------------------------------------------------------
// --------------------------------------------------------------
// --------------------------------------------------------------
// Implementation
template <class Super>
void EnvGrid3DPlugIn<Super>::install()
{
	m_Pos = new BehaveRT::DeviceArrayWrapper<float4>(
		m_CommonRes.getDeviceInterface(), 
		hBody3DParams.numBodies, 
		1, true, true);

	addToFeaturesToReorder(m_Pos, m_Pos->getType());

	m_Pos->bindToField(hBody3DFields.position);

	Body3D::Body3D_copyFieldsToDevice();
	
	m_AgentHash = new BehaveRT::DeviceArrayWrapper<uint2>(
		m_CommonRes.getDeviceInterface(), 
		hBody3DParams.numBodies);

	
	m_AgentHash->bindToField(hEnvGrid3DFields.hash);

	// ------

	read_config_param(EnvGrid3D, cellSize, Float3);
	read_config_param(EnvGrid3D, disableSorting, Bool);
	read_config_param(EnvGrid3D, worldCenter, Float3);
	read_config_param(EnvGrid3D, lockWorldProportions, Bool);

	setupWorldRadius();

	setupGrid();

	setupMaxBodiesPerCell();

	char msg[100];
	sprintf(msg, "%d x %d x %d = %d cells\n", 
		hEnvGrid3DParams.gridSize.x, hEnvGrid3DParams.gridSize.y, 
		hEnvGrid3DParams.gridSize.z, hEnvGrid3DParams.numCells);
	m_CommonRes.getLogger()->log("EnvGrid3D", msg);

	sprintf(msg, "WorldRadius: %f x %f x %f\n", 
		hEnvGrid3DParams.worldRadius.x, hEnvGrid3DParams.worldRadius.y, 
		hEnvGrid3DParams.worldRadius.z);
	m_CommonRes.getLogger()->log("EnvGrid3D", msg);

	sprintf(msg, "WorldCenter: %f x %f x %f\n", 
		hEnvGrid3DParams.worldCenter.x, hEnvGrid3DParams.worldCenter.y, 
		hEnvGrid3DParams.worldCenter.z);
	m_CommonRes.getLogger()->log("EnvGrid3D", msg);
	
	sprintf(msg, "cellSize: %f %f %f [bodies/cell: %d]\n", hEnvGrid3DParams.cellSize.x,
		hEnvGrid3DParams.cellSize.y, hEnvGrid3DParams.cellSize.z, 
		hEnvGrid3DParams.maxBodiesPerCell);
	m_CommonRes.getLogger()->log("EnvGrid3D", msg);

	EnvGrid3D::EnvGrid3D_copyFieldsToDevice();

}

// --------------------------------------------------------------

template <class Super>
void EnvGrid3DPlugIn<Super>::uninstall()
{
	delete m_Pos;
	delete m_AgentHash;
}

// --------------------------------------------------------------

template <class Super>
void EnvGrid3DPlugIn<Super>::reset()
{
	Super::reset(); // MANDATORY OPERATION

	// Initi the arrays
	for (BehaveRT::uint i = 0; i < hBody3DParams.numBodies; i ++ )
	{
		//float3 unitRand = unitRandFloat3();
		//float multipler = 1 - frand() * frand();

		float3 unitRand = make_float3(
			frand() - 0.5, frand() - 0.5, frand() - 0.5);
		float multipler = 2;

		float4 initPos = make_float4(
			(unitRand.x) * hEnvGrid3DParams.worldRadius.x * multipler + hEnvGrid3DParams.worldCenter.x, 
			(unitRand.y) * hEnvGrid3DParams.worldRadius.y * multipler + hEnvGrid3DParams.worldCenter.y, 
			(unitRand.z) * hEnvGrid3DParams.worldRadius.z * multipler + hEnvGrid3DParams.worldCenter.z, 
			1);

		m_Pos->setHostArrayElement(i, &initPos);
	}

	m_Pos->copyArrayToDevice();
	m_Pos->swapPosReadWrite();

	// Initialize arrays
	for (BehaveRT::uint i = 0; i < hBody3DParams.numBodies; i ++ )
	{
		uint2 initHash = make_uint2(1, i);
		m_AgentHash->setHostArrayElement(i, &initHash);
	}

	m_AgentHash->copyArrayToDevice();
	m_AgentHash->swapPosReadWrite();

	// --------------------------------------

	m_ParamListGL->AddParam(
		new Param<float>("World radius", 
			hEnvGrid3DParams.worldRadius.x, 
			2.0, 100.0, 2.0, 
			&hEnvGrid3DParams.worldRadius.x));

	//m_ParamListGL->AddParam(
	//	new Param<float>("Cell size", 
	//		hEnvGrid3DParams.cellSize.x, 
	//		0.5, 100.0, 0.5, 
	//		&hEnvGrid3DParams.cellSize.x));

}

// --------------------------------------------------------------

template <class Super>
void EnvGrid3DPlugIn<Super>::update(const float elapsedTime)
{
	Super::update(elapsedTime); // MANDATORY OPERATION

	// Insert here the default update operation
}

// --------------------------------------------------------------
// --------------------------------------------------------------

//#define DEBUG_REORDERING 1

template <class Super>
void EnvGrid3DPlugIn<Super>::reorderSimData()
{
	assert(m_CommonRes.isInitialized());

	if (hEnvGrid3DParams.lockWorldProportions)
	{
		// Any change to X is propagated to Y and Z values
		hEnvGrid3DParams.worldRadius.y = 
				hEnvGrid3DParams.worldRadius.z = hEnvGrid3DParams.worldRadius.x;
		hEnvGrid3DParams.cellSize.y = 
				hEnvGrid3DParams.cellSize.y = hEnvGrid3DParams.cellSize.z;
	}
	
	m_CommonRes.getDeviceInterface()->kernelCall(
		hBody3DParams.numBodies, 256,
		EnvGrid3D::genericCalcHashDRef(), 
		&EnvGrid3D::EnvGrid3D_beforeKernelCall,
		&EnvGrid3D::EnvGrid3D_afterKernelCall);

	m_AgentHash->swapPosReadWrite();

	if (hEnvGrid3DParams.disableSorting) 
		return;

    // sort agents based on hash
	BehaveRT::RadixSort((KeyValuePair *) m_AgentHash->getReadDeviceArray(), 
		(KeyValuePair *) m_AgentHash->getWriteDeviceArray(), 
		hBody3DParams.numBodies, 32);

	m_CommonRes.getDeviceInterface()->threadSync();

	// reorder agent arrays into sorted order and
	// find start of each cell

	int envolvedFields[2];
	envolvedFields[0] = hEnvGrid3DFields.hash;
	envolvedFields[1] = hEnvGrid3DFields.featureToReorder;

	

	int i = 0;
	// Iterate the list of installed array wrappers
	for (FeaturesToReorder::const_iterator it = m_FeaturesToReorder.begin();
		it != m_FeaturesToReorder.end(); it ++)
	{
		
		i ++;
		FeatureToReorder feature = (FeatureToReorder) *it;

		if (feature.second == &typeid(float4))
		{
			DeviceArrayWrapper<float4>* arrayWrapper = (DeviceArrayWrapper<float4>*) feature.first;
			arrayWrapper->bindToField(hEnvGrid3DFields.featureToReorder);
			envolvedFields[1] = hEnvGrid3DFields.featureToReorder;
			
			

#ifdef DEBUG_REORDERING
			// DEBUG PRINT
			printf("----\n POS     %d)\n", i);
			printf("REORDER FLOAT4 %d\n", hEnvGrid3DFields.featureToReorder);
			printf("REORDER FLOAT4 [%d] %d\n", i, envolvedFields[1]);

			char msg[100];
			sprintf(msg, "COUNT %d\n", i);
			m_CommonRes.getLogger()->log(name(), msg);
#endif

			//if (hEnvGrid3DFields.featureToReorder != 0)
			//if (false)
			//{// DEBUG -IF_SECTION- invalid device pointer (10-03-10 RESOLVED)
				// Error with == 0, != 0
				
				EnvGrid3D::EnvGrid3D_copyFieldsToDevice();
				
				m_CommonRes.getDeviceInterface()->kernelCallUsingFields(
					envolvedFields, 2, 
					hBody3DParams.numBodies, 256,
					//EnvGrid3D::genericReorderDataRef_float4(), 
					EnvGrid3D::reorderDataFloat4Ref(), 
					&EnvGrid3D::EnvGrid3D_beforeKernelCallSimple,
					&EnvGrid3D::EnvGrid3D_afterKernelCallSimple);
				
				arrayWrapper->swapPosReadWrite();
			//} // DEBUG END
		}
		else if (feature.second == &typeid(BehaveRT::uint))
		{
			DeviceArrayWrapper<BehaveRT::uint>* arrayWrapper = 
				(DeviceArrayWrapper<BehaveRT::uint>*) feature.first;
			arrayWrapper->bindToField(hEnvGrid3DFields.featureToReorder);
			envolvedFields[1] = hEnvGrid3DFields.featureToReorder;

			m_CommonRes.getDeviceInterface()->kernelCallUsingFields(
				envolvedFields, 2, 
				hBody3DParams.numBodies, 256,
				EnvGrid3D::genericReorderDataRef_uint(), 
				&EnvGrid3D::EnvGrid3D_beforeKernelCall,
				&EnvGrid3D::EnvGrid3D_afterKernelCall);

			//arrayWrapper->swapPosReadWrite();
		}
		else if (feature.second == &typeid(float))
		{
			DeviceArrayWrapper<BehaveRT::uint>* arrayWrapper = 
				(DeviceArrayWrapper<BehaveRT::uint>*) feature.first;
			arrayWrapper->bindToField(hEnvGrid3DFields.featureToReorder);
			envolvedFields[1] = hEnvGrid3DFields.featureToReorder;

			m_CommonRes.getDeviceInterface()->kernelCallUsingFields(
				envolvedFields, 2, 
				hBody3DParams.numBodies, 256,
				EnvGrid3D::genericReorderDataRef_float(), 
				&EnvGrid3D::EnvGrid3D_beforeKernelCall,
				&EnvGrid3D::EnvGrid3D_afterKernelCall);

			//arrayWrapper->swapPosReadWrite();
		}
	}

	m_CommonRes.getDeviceInterface()->threadSync();
	
} // reorderSimData


// ------------------------------------------------------------

template <class Super>
void EnvGrid3DPlugIn<Super>::setupWorldRadius()
{
	if (!BehaveRT::StringConverter::parseBool(
		m_CommonRes.getConfig()->getSetting("keepDensity", EnvGrid3DPlugIn::name())))
	{
		//hEnvGrid3DParams.worldRadius =
		//	BehaveRT::StringConverter::parseFloat(m_CommonRes.getConfig()->getSetting("worldRadius", "EnvGrid3DPlugIn"));
		read_config_param(EnvGrid3D, worldRadius, Float3);
		if (hEnvGrid3DParams.worldRadius.x == 0.0)
		{
			hEnvGrid3DParams.worldRadius.x = BehaveRT::StringConverter::parseFloat(
				m_CommonRes.getConfig()->getSetting(
					"worldRadius", EnvGrid3DPlugIn::name()));
			hEnvGrid3DParams.worldRadius.y = 
				hEnvGrid3DParams.worldRadius.z = hEnvGrid3DParams.worldRadius.x;
		}
		return;
	}
	
	// Keep constant density
	float density;
	if (hBody3DParams.use2DProjection)
	{
		density = BehaveRT::StringConverter::parseFloat(
			m_CommonRes.getConfig()->getSetting("bodiesDensity2D", EnvGrid3DPlugIn::name()));
		
		// Keep circular area density
		//hEnvGrid3DParams.worldRadius = 
		//	pow( hBody3DParams.numBodies / ( 3.14159 * density ), 0.5 );

		hEnvGrid3DParams.worldRadius.x = 
			pow( hBody3DParams.numBodies / density, 0.5f ) / 2;	

		hEnvGrid3DParams.worldRadius.y = 
				hEnvGrid3DParams.worldRadius.z = hEnvGrid3DParams.worldRadius.x;

	}
	else
	{
		density = BehaveRT::StringConverter::parseFloat(
			m_CommonRes.getConfig()->getSetting("bodiesDensity3D", EnvGrid3DPlugIn::name()));

		// Keep spherical volume density
		//hEnvGrid3DParams.worldRadius = 
		//	pow( 3 * hBody3DParams.numBodies / ( 4 * 3.14159 * density ), 0.33333 );
		
		hEnvGrid3DParams.worldRadius.x = 
			pow( hBody3DParams.numBodies / density, 0.33333f ) / 2;		

		hEnvGrid3DParams.worldRadius.y = 
				hEnvGrid3DParams.worldRadius.z = hEnvGrid3DParams.worldRadius.x;
	}
} // setupWorldRadius

// ------------------------------------------------------------


template <class Super>
float3 EnvGrid3DPlugIn<Super>::getAugmentedRadius( float3 worldRadius, float3 cellSize )
{
	return make_float3( 
		worldRadius.x + cellSize.x * 2,
		worldRadius.y + cellSize.y * 2,
		worldRadius.z + cellSize.z * 2);
}

template <class Super>
void EnvGrid3DPlugIn<Super>::setupGrid()
{
	float3 augmentedWorldRadius = 
		getAugmentedRadius(hEnvGrid3DParams.worldRadius, hEnvGrid3DParams.cellSize);

	hEnvGrid3DParams.gridSize = make_uint3(
		augmentedWorldRadius.x * 2 / hEnvGrid3DParams.cellSize.x,
		augmentedWorldRadius.y * 2 / hEnvGrid3DParams.cellSize.y,
		augmentedWorldRadius.z * 2 / hEnvGrid3DParams.cellSize.z);

	hEnvGrid3DParams.worldOrigin = make_float3(
		hEnvGrid3DParams.worldCenter.x - augmentedWorldRadius.x, 
		hEnvGrid3DParams.worldCenter.y - augmentedWorldRadius.y,
		hEnvGrid3DParams.worldCenter.z - augmentedWorldRadius.z);

	hEnvGrid3DParams.numCells = 
		hEnvGrid3DParams.gridSize.x * 
		hEnvGrid3DParams.gridSize.y * 
		hEnvGrid3DParams.gridSize.z;
    
} // setupGrid

// --------------------------------------------------------------

template <class Super>
void EnvGrid3DPlugIn<Super>::setWorldCenter( float3 worldCenter )
{
	float augmentedWorldRadius = getAugmentedRadius(hEnvGrid3DParams.worldRadius, hEnvGrid3DParams.cellSize.x);

	hEnvGrid3DParams.worldOrigin = make_float3(
		-augmentedWorldRadius + hEnvGrid3DParams.worldCenter.x, 
		-augmentedWorldRadius + hEnvGrid3DParams.worldCenter.y, 
		-augmentedWorldRadius + hEnvGrid3DParams.worldCenter.z);
}

template <class Super>
float3 EnvGrid3DPlugIn<Super>::getWorldCenter()
{
	float3 augmentedWorldRadius = getAugmentedRadius(hEnvGrid3DParams.worldRadius, hEnvGrid3DParams.cellSize);
	return make_float3(
		hEnvGrid3DParams.worldOrigin.x + augmentedWorldRadius.x, 
		hEnvGrid3DParams.worldOrigin.y + augmentedWorldRadius.y, 
		hEnvGrid3DParams.worldOrigin.z + augmentedWorldRadius.z);
}


// --------------------------------------------------------------

template <class Super>
void EnvGrid3DPlugIn<Super>::setupMaxBodiesPerCell()
{
	if (!BehaveRT::StringConverter::parseBool(
		m_CommonRes.getConfig()->getSetting("useAutomaticMaxBodiesPerCell", EnvGrid3DPlugIn::name())))
	{
		//hEnvGrid3DParams.maxBodiesPerCell =
		//	BehaveRT::StringConverter::parseInt(m_CommonRes.getConfig()->getSetting("maxBodiesPerCell", "EnvGrid3DPlugIn"));

		read_config_param(EnvGrid3D, maxBodiesPerCell, Int);
		return;
	}

	// Calculate the upper bound of number of body per cell 
	float cellVolume = hEnvGrid3DParams.cellSize.x *
		hEnvGrid3DParams.cellSize.y *
		hEnvGrid3DParams.cellSize.z;

	float bodyVolume = 4.1866f * 
		hBody3DParams.commonRadius * 
		hBody3DParams.commonRadius * 
		hBody3DParams.commonRadius; // Volume of a Sphere: PI*4/3*R^3
	
	// Calculcate as UPPER BOUND, rounding up to
	hEnvGrid3DParams.maxBodiesPerCell = (int) (1 + cellVolume / bodyVolume);

} // setupMaxBodiesPerCell

// --------------------------------------------------------------

template <class Super>
void EnvGrid3DPlugIn<Super>::addToFeaturesToReorder(InstallableEntity* entity, const type_info* type)
{
	entity->setInstallationKey(m_FeaturesToReorder.size());
	m_FeaturesToReorder.push_back(std::make_pair(entity, type));
}